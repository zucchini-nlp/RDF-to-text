#!/bin/bash

python ./dbgpt_hub/utils/preprocess_dataset.py 

python train_qlora.py \
    --dataset_name lcquad \
    --instruction_template zeroshot_triplets \
    --model_name_or_path lmsys/vicuna-13b-v1.3 \
    --output_dir adapters_qlora \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --max_steps 300 \
    --learning_rate 2e-4 \
    --optim "adamw_hf" \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj;v_proj" \
    --logging_steps 10 \
    --warmup_steps 10 \
    --do_train True \
    --do_eval True \
    --gradient_checkpointing True

echo "finished"

