export num_gpus=8
export output_dir="outputs/e2e_gptm"
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=$num_gpus run_glue.py \
CUDA_VISIBLE_DEVICES=2 python examples/run_generation.py \
--model_name_or_path gpt2-m \
--model_type gpt2 \
--dataset_name e2e_nlg \
--do_train \
--do_eval \
--deepspeed "dsconfig/zero0_config.json" \
--padding_side "left" \
--group_by_length \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 8 \
--save_strategy "steps" \
--evaluation_strategy "steps" \
--eval_steps 50 \
--save_steps 50 \
--learning_rate 5e-5 \
--lr_scheduler_type "linear" \
--pad_to_max_length \
--max_eval_samples 2000 \
--model_max_length 512 \
--num_train_epochs 5 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--warmup_ratio 0.0  \
--num_beams 10 \
--seed 0 \
--fp16 \
--weight_decay 0.0 \
--load_best_model_at_end \
--weight_decay 0
