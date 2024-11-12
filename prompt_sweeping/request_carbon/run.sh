#!/bin/bash
set -x

SEED=0
MODEL=$2 #"13b-chat"
PROMPT=$1 # "math_all_brief+"

python vllm_sweep_prompt.py --prompt_file ../prompts/prompt_${PROMPT}.json \
    --model_name /work/li.baol/llama_models/llama-2-${MODEL}-hf \
    --output_file ${PROMPT}_${MODEL} --dtype bfloat16
