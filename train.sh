#!/bin/bash
# query_key_value

python ./src/utils/preprocess_dataset.py 

CUDA_VISIBLE_DEVICES="1,2,3" python train_qlora.py \
    --dataset_name vquanda \
    --instruction_template zeroshot_question \
    --model_name_or_path EleutherAI/gpt-j-6B \
    --cache_dir "/archive/turganbay/.huggingface" \
    --output_dir adapters_qlora \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --max_steps 150 \
    --learning_rate 2e-4 \
    --optim "adamw_hf" \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj;v_proj" \
    --logging_steps 10 \
    --warmup_steps 10 \
    --do_train True \
    --do_eval True

echo "finished"

