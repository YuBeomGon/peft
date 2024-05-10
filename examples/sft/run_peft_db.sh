#!/bin/bash

model_name="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
table_names="alignment_table,chat_table"
logging_steps=1000

python peft/examples/sft/train.py \
--seed 100 \
--model_name_or_path $model_name \
--database_table_name $table_names \
--database_url "postgresql://postgres:kk3249@localhost?port=5432&dbname=alignment" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 4096 \
--report_to="wandb" \
--num_train_epochs 1 \
--logging_steps $logging_steps \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "steps" \
--eval_steps $logging_steps \
--logging_steps $logging_steps \
--save_steps $logging_steps \
--warmup_steps 500 \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "outputs/"$model_name \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing True \
--use_reentrant True \
--dataset_text_field "content" \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--use_flash_attn True
