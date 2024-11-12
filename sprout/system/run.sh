#!/bin/bash

set -x 
START_HR=$1

# for START_HR in 3900 7000
# do
for REGION in "NL" "AU-SA" "GB" "US-TEX" "US-CAL"
do
    python main.py --region $REGION --optimizer_name baseline --pref_torelance 0.1 --start_hour $START_HR
done
# done

echo "done!"