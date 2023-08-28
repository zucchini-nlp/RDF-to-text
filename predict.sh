#!/bin/bash

python ./dbgpt_hub/utils/preprocess_dataset.py 

CUDA_VISIBLE_DEVICES=1 python predict_qlora.py \
    --model_name_or_path lmsys/vicuna-13b-v1.3 \
    --peft_ckpt_path "./adapter/checkpoint-50/" \
    --dataset_name web_nlg \
    --output_name "data/webnlg_preds_{model_name_or_path}_{time_now}.json"

echo "finished"
