import json
import os

# Input file path
input_file = "data/thesis_ocr_output/processed_combined_json.json"

# Output file path
#output_file = "data/thesis_ocr_output/prepared_data.json"


output_dir = "data/thesis_ocr_output"
os.makedirs(output_dir, exist_ok=True)

# Define OCR systems to extract
ocr_systems = {
    "tesseract": "tesseract_v5.5.0_new_swebest_oem1_psm6",
    "doctr": "doctr", 
    "idefics": "idefics",
    "kraken": "kraken-german"
}

# Simply save the original file to make it accessible for run.py
# This approach keeps changes minimal while allowing run.py to handle the extraction logic
output_file = os.path.join(output_dir, "processed_combined_json.json")

# Just copying the file to the expected location if it's not already there
if os.path.abspath(input_file) != os.path.abspath(output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

print(f"File prepared for OCR processing: {output_file}")
print(f"OCR systems that will be processed: {', '.join(ocr_systems.keys())}")