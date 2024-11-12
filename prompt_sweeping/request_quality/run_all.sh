#!/bin/bash
set -x

# ./run_beam.sh math_all 13b-chat &&
#./run.sh math_all 7b-chat &&
#./run.sh math_all_brief 13b-chat &&
#./run.sh math_all_brief+ 13b-chat &&

# ./run_beam.sh alpaca_5000 13b-chat &&
#./run.sh alpaca_5000 7b-chat &&
#./run.sh alpaca_5000_brief 13b-chat &&
#./run.sh alpaca_5000_brief+ 13b-chat &&

# ./run_beam.sh mmlu_5000 13b-chat &&
#./run.sh mmlu_5000 7b-chat &&
#./run.sh mmlu_5000_brief 13b-chat &&
#./run.sh mmlu_5000_brief+ 13b-chat &&

./run.sh scienceqa_5000 13b-chat &&
./run.sh scienceqa_5000 7b-chat &&
./run.sh scienceqa_5000_brief 13b-chat &&
./run.sh scienceqa_5000_brief+ 13b-chat &&

# ./run.sh helpful_all 7b-chat &&
# ./run.sh helpful_all_brief 13b-chat &&
# ./run.sh helpful_all 13b-chat &&
# ./run.sh helpful_all_brief+ 13b-chat &&
# ./run_beam.sh mmlu_5000 13b-chat &&
# ./run_beam.sh math_all 13b-chat &&
# ./run_beam.sh helpful_all 13b-chat &&

echo "done!"
