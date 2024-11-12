#!/bin/bash

set -x

mkdir -p assessment_results/test_10

rm -f /work/li.baol/GIT/alpaca_eval/src/alpaca_eval/evaluators_configs/gpt4/*.json
alpaca_eval evaluate \
    --input_path="sample_10" \
    --output_path="assessment_results/test_10/details_2000.json" \
    --summary_path="assessment_results/test_10/summary_2000.json" \
    --annotators_config="gpt4" && 

echo "done!"