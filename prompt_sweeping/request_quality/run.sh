#!/bin/bash
set -x

MODEL=$2 #"13b-chat"
PROMPT=$1 # "math_all"

python vllm_batched_inference_chat.py --prompt_file ../prompts/prompt_${PROMPT}.json \
    --model_name /work/li.baol/llama_models/llama-2-${MODEL}-hf \
    --save_path ${PROMPT}_${MODEL} --dtype float16
