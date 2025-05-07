echo "------- Running post-OCR pipeline -------"

# Load arguments from JSON
ARGS=$(jq -r 'to_entries | map("--" + .key + " " + (.value|tostring)) | join(" ")' configs/my_args.json)

echo "Preparing data for post-OCR pipeline..."
# Run the Python script with the loaded arguments
python scripts/prepare_json_for_post_ocr.py

echo "Running post-OCR"
python scripts/run.py $ARGS