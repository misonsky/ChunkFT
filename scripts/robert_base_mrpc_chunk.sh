export num_gpus=1
export output_dir="outputs/mrpc"
port=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES="0" torchrun --master_port "$port" --nproc_per_node=$num_gpus examples/run_glue.py \
--model_name_or_path roberta-base \
--task_name mrpc \
--do_train \
--do_eval \
--do_predict \
--max_seq_length 512 \
--per_device_train_batch_size 64 \
--learning_rate 3e-5 \
--num_train_epochs 50 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0 \
--eval_metric "metrics/accuracy" \
--seed 0 \
--chunk_tuning \
--weight_decay 0 \
--chunk_num $1 \
--chunk_strategy $2 \
--load_best_model_at_end