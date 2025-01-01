export num_gpus=1
export output_dir="outputs/mnli_o"
port=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=2 torchrun --master_port "$port" --nproc_per_node=$num_gpus examples/run_glue.py \
--model_name_or_path roberta-base \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 1e-05 \
--num_train_epochs 10 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 100 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0