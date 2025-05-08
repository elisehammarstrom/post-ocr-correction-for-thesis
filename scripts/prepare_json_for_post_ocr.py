import json

# Input file path
#input_file = "data/thesis_ocr_output/processed_old_json.json"
input_file = "data/thesis_ocr_output/processed_combined_json.json"

# Output file path
output_file = "data/thesis_ocr_output/prepared_data.json"

# List to store transformed data
prepared_data = []

# Open and process the input JSON file
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        # Parse each line as JSON
        data = json.loads(line)
        
        # Extract relevant fields
        transcription = data.get("transcription", "").lstrip("\ufeff").strip()  # Ground truth
        ocr = data.get("ocr", {}).get("tesseract_v5.5.0_new_swebest_oem1_psm6", "").strip()  # OCR output
        #ocr = data.get("ocr", {}).get("Tesseract_v5.5.0", "").strip()  # OCR output
        
        # Append transformed data
        prepared_data.append({
            "file": data.get("id", ""),
            "gt": transcription,
            "ocr": ocr
        })

# Save the transformed data to a new JSON file
with open(output_file, "w", encoding="utf-8") as out_file:
    json.dump(prepared_data, out_file, ensure_ascii=False, indent=4)

print(f"Prepared data saved to {output_file}")