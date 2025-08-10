#!/bin/bash

export OUTPUT_DIR="../../output"
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export DAY="3day"
export SET_TYPE="3day_gpt4o_orig"
export STRATEGY="direct_og"
export CSV_FILE="../../final_3day_dataset.csv"

cd tools/planner

python sole_planning_mltp.py \
    --day $DAY \
    --set_type $SET_TYPE \
    --output_dir $OUTPUT_DIR \
    --csv_file $CSV_FILE \
    --model_name $MODEL_NAME \
    --strategy $STRATEGY