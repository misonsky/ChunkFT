# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
import numpy as np
import importlib.util
import math
import time
import random
import re
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.distributed as dist
from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput, speed_metrics,ShardedDDPOption,PREFIX_CHECKPOINT_DIR
from transformers.trainer_pt_utils import get_parameter_names,reissue_pt_warnings
from transformers.utils import is_sagemaker_mp_enabled
ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

def is_fairscale_available():
    return importlib.util.find_spec("fairscale") is not None

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, 
                eval_examples=None, 
                post_process_function=None, 
                num_group=1,
                optimizer_strategy="down2up",
                keep_position=None,
                keeping_layers=None,
                layer_names = None,
                freeze_emb = False,
                freeze_output = False,
                random_tuning=False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.optimizer_states = None
        self.counter = 0
        self.num_group=1
        self.freeze_emb = freeze_emb
        self.strategy = optimizer_strategy
        self.random_tuning = random_tuning
        self.layer_names = layer_names
        self.keep_position = keep_position
        self.keeping_layers = keeping_layers
        self.freeze_emb = freeze_emb
        self.freeze_output = freeze_output
        self.init_group_parameters()
    def init_group_parameters(self):
        if self.strategy=="down2up":
            if self.keep_position is None:
                self.group_parameters = [self.layer_names[0]] + [str(i) for i in range(self.model.config.num_hidden_layers)] + [self.layer_names[1]]
            elif self.keep_position == "top":
                self.group_parameters = [self.layer_names[0]] + [str(i) for i in range(self.model.config.num_hidden_layers)]
            elif self.keep_position == "down":
                self.group_parameters = [str(i) for i in range(self.model.config.num_hidden_layers)] + [self.layer_names[1]]
            else:
                self.group_parameters = [str(i) for i in range(self.model.config.num_hidden_layers)]
        elif self.strategy=="up2down":
            if self.keep_position is None:
                self.group_parameters = [self.layer_names[1]] + [str(i) for i in range(self.model.config.num_hidden_layers-1,-1,-1)] + [self.layer_names[0]]
            elif self.keep_position == "top":
                self.group_parameters = [str(i) for i in range(self.model.config.num_hidden_layers-1,-1,-1)] + [self.layer_names[0]]
            elif self.keep_position == "down":
                self.group_parameters = [self.layer_names[1]] + [str(i) for i in range(self.model.config.num_hidden_layers-1,-1,-1)]
            else:
                self.group_parameters = [str(i) for i in range(self.model.config.num_hidden_layers-1,-1,-1)]
        else:
            if self.keep_position is None:
                self.group_parameters = [self.layer_names[0]] + [str(i) for i in range(self.model.config.num_hidden_layers)] + [self.layer_names[1]]
            elif self.keep_position == "top":
                self.group_parameters = [self.layer_names[0]] + [str(i) for i in range(self.model.config.num_hidden_layers)]
            elif self.keep_position == "down":
                self.group_parameters = [str(i) for i in range(self.model.config.num_hidden_layers)] + [self.layer_names[1]]
            else:
                self.group_parameters = [str(i) for i in range(self.model.config.num_hidden_layers)]
            random.shuffle(self.group_parameters)
        if self.freeze_emb and self.keep_position != "down" and self.keep_position != "top-down":
            self.group_parameters.remove(self.layer_names[0])
        if self.freeze_output and self.keep_position != "top" and self.keep_position != "top-down":
            self.group_parameters.remove(self.layer_names[1])
    def init_layers(self,name):
        if self.layer_names[-1] in name:
            return True
        return False
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and self.init_layers(n))
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and self.init_layers(n)) 
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum(dict((p.data_ptr(), p.numel()) for p in module.parameters()).values())
                            print(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    def select_element(self):
        elements = self.group_parameters[:self.num_group]
        for element in elements:
            self.group_parameters.remove(element)
        for element in elements:
            self.group_parameters.append(element)
        if len(self.keeping_layers)>0:
            elements.extend(self.keeping_layers)
        return elements
    def pattern_name(self):
        pattern = None
        if self.freeze_emb and self.freeze_output:
            pattern = rf'\d+'
        elif self.freeze_emb:
            pattern = rf'\d+|\b(?:{self.layer_names[1]})\b'
        elif self.freeze_output:
            pattern = rf'\d+|\b(?:{self.layer_names[0]})\b'
        else:
            pattern = rf'\d+|\b(?:{self.layer_names[0]}|{self.layer_names[1]})\b'
        return pattern

    def update_parameter_state(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if int(self.counter // self.num_group) == int(self.model.config.num_hidden_layers // self.num_group):
            if self.strategy == "random":
                random.shuffle(self.group_parameters)
            self.counter = 0
            self.optimizer_states=self.get_optimizer_state()
        pattern = self.pattern_name()
        elements = self.select_element()
        # print("select {}".format(" ".join(elements)))
        self.counter +=1
        #select parameters
        for name,parameter in opt_model.named_parameters():
            parameter.requires_grad = False
            names_sel = re.findall(pattern, name)
            if len(names_sel)>0 and str(names_sel[0]) in elements:
                parameter.requires_grad = True
        # self.check_trainable()
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad) 
                    ]
            },
            ]
        assert len(self.optimizer_states) == len(optimizer_grouped_parameters)
        for i in range(len(optimizer_grouped_parameters)):
            optimizer_grouped_parameters[i].update(self.optimizer_states[i])
        self.optimizer.param_groups = optimizer_grouped_parameters
        # print("selecting layer number is {}".format(element))
    def get_optimizer_state(self):
        states_groups={}
        optimizer_grouped_parameters = self.optimizer.param_groups
        for i, param_group in enumerate(optimizer_grouped_parameters):
            param_group.pop("params")
            states_groups[i] = param_group
        return states_groups
    def check_trainable(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        for n, p in opt_model.named_parameters():
            if p.requires_grad:
                print(n)
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.optimizer_states is None and self.random_tuning:
            self.optimizer_states=self.get_optimizer_state()
        if self.random_tuning:
            self.update_parameter_state()
        # trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)#334094338 # 44381186 #12598274
        # print(f"Trainable parameters: {trainable_parameters}")
        # self.check_trainable()
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        return loss.detach()
    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        start_time = time.time()
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        if self.post_process_function is not None and self.compute_metrics is not None and self.args.should_save:
            # Only the main node write the results by default
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
            metrics.update(output.metrics)
        else:
            metrics = output.metrics

        if self.args.should_log:
            # Only the main node log the results by default
            self.log(metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        start_time = time.time()
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        metrics.update(output.metrics)
        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            # torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            # with warnings.catch_warnings(record=True) as caught_warnings:
            #     torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            # reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)