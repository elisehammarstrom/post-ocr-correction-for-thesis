#!/bin/bash

echo "------- Running post-OCR pipeline -------"

echo "Preparing data for post-OCR pipeline..."
# Run the Python script with the loaded arguments
python scripts/prepare_json_for_post_ocr.py

echo "Running post-OCR"
# Explicitly specify the model path and point to the config file
python scripts/run.py --model_name_or_path google/byt5-small --config_name configs/my_args.json