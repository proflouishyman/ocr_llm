#!/bin/bash
#backs up the train.py file to the current directory


# Define the source file path
source_file="/home/lhyman6/.local/lib/python3.8/site-packages/llava/train/train.py"

# Check if the source file exists
if [ -f "$source_file" ]; then
    # Move the file to the current directory
    cp "$source_file" .
    echo "File moved successfully."
else
    echo "Source file does not exist."
fi
