#!/bin/bash

# Directory paths
BASE_DIR="/scratch4/lhyman6/OCR/OCR/ocr_llm/work"
SCRIPT_PATH="/data/lhyman6/OCR/scripts/train_bart_template.sh"

# Dataset names and sizes
datasets=("gold" "silver")
sizes=(100 1000 10000)

# Loop through each dataset and size
for dataset in "${datasets[@]}"; do
    for size in "${sizes[@]}"; do
        TRAIN_DATASET="${BASE_DIR}/${dataset}_${size}/train"
        VALIDATION_DATASET="${BASE_DIR}/${dataset}_${size}/val"
        TEST_DATASET="${BASE_DIR}/${dataset}_${size}/test"
        OUTPUT_DIR="${BASE_DIR}/tuning_results_${dataset}_${size}"

        # Print the details of the job being submitted
        echo "Submitting job for ${dataset}_${size}"
        echo "Train dataset: ${TRAIN_DATASET}"
        echo "Validation dataset: ${VALIDATION_DATASET}"
        echo "Test dataset: ${TEST_DATASET}"
        echo "Output directory: ${OUTPUT_DIR}"

        # Submit the SLURM job
        sbatch $SCRIPT_PATH $TRAIN_DATASET $VALIDATION_DATASET $TEST_DATASET $OUTPUT_DIR
    done
done
