#!/bin/bash

set -x

rm -f /home/ubuntu/GIT/alpaca_eval/src/alpaca_eval/evaluators_configs/scienceqa_gpt4/annotations_seed0_configs.json

alpaca_eval evaluate \
    --model_outputs="scienceqa/samples_800_seed_0/scienceqa_5000_brief_13b.json" \
    --reference_outputs="scienceqa/samples_800_seed_0/correct_answer.json" \
    --annotators_config="scienceqa_gpt4" --output_path scienceqa/result_brief_vs_dataset && 

alpaca_eval evaluate \
    --model_outputs="scienceqa/samples_800_seed_0/scienceqa_5000_brief+_13b.json" \
    --reference_outputs="scienceqa/samples_800_seed_0/correct_answer.json" \
    --annotators_config="scienceqa_gpt4" --output_path scienceqa/result_brief+_vs_dataset && 

alpaca_eval evaluate \
    --model_outputs="scienceqa/samples_800_seed_0/scienceqa_5000_13b.json" \
    --reference_outputs="scienceqa/samples_800_seed_0/correct_answer.json" \
    --annotators_config="scienceqa_gpt4" --output_path scienceqa/result_13b_vs_dataset && 

# alpaca_eval evaluate \
#     --model_outputs="scienceqa/samples_800_seed_0/scienceqa_5000_7b.json" \
#     --reference_outputs="scienceqa/samples_800_seed_0/correct_answer.json" \
#     --annotators_config="scienceqa_gpt4" --output_path scienceqa/result_7b_vs_dataset && 

echo "done!"