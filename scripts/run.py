from functools import partial
import jiwer
import json

from post_ocr.data import chunkify, filter_bad_sample, segment
from post_ocr.arguments import get_args
from post_ocr.training import run_training
from datasets import Dataset, DatasetDict, concatenate_datasets

import os
import warnings

import tensorflow as tf

def prepare_data_from_json(json_path, args, test_fns=None, val_fns=None, seed=666):
    """Load data from a JSON file instead of kubhist dataset"""
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to dataset
    ds = Dataset.from_dict({
        "file": [item["file"] for item in data],
        "gt": [item["gt"] for item in data],
        "ocr": [item["ocr"] for item in data]
    })
    
    split_size = 0.15
    
    # Handle dataset splitting with same logic as original
    if test_fns is None and val_fns is None:
        train_val_test = ds.train_test_split(test_size=split_size, seed=seed)
        test = train_val_test["test"]
        train_val = train_val_test["train"].train_test_split(test_size=split_size, seed=seed)
        train = train_val["train"]
        val = train_val["test"]
    elif test_fns is not None:
        test = ds.filter(lambda x: x["file"].split("/")[-1] in test_fns)
        if val_fns is not None:
            val = ds.filter(lambda x: x["file"].split("/")[-1] in val_fns)
            train = ds.filter(
                lambda x: x["file"].split("/")[-1] not in val_fns
                and x["file"].split("/")[-1] not in test_fns
            )
        else:
            train_val = ds.filter(
                lambda x: x["file"].split("/")[-1] not in test_fns
            ).train_test_split(test_size=split_size, seed=seed)
            train = train_val["train"]
            val = train_val["test"]
    
    print("Original data setup with one file per item")
    print(train)
    print(val)
    print(test)
    
    # Apply the same processing steps as the original
    my_filter_bad_sample = partial(filter_bad_sample, min_len=4, max_cer=0.5)
    
    # Create aligned versions with same processing as original
    my_segment = partial(segment, align=True, max_length=128, max_cer=0.8)
    
    test_chunks_aligned = (
        test.map(my_segment, batched=False, num_proc=4)
        .filter(my_filter_bad_sample)
        .map(chunkify, batched=True, num_proc=4)
    )
    val_chunks_aligned = (
        val.map(my_segment, batched=False, num_proc=4)
        .filter(my_filter_bad_sample)
        .map(chunkify, batched=True, num_proc=4)
    )
    train_chunks_aligned = (
        train.map(my_segment, batched=False, num_proc=4)
        .filter(my_filter_bad_sample)
        .map(chunkify, batched=True, num_proc=4)
    )
    
    print("Segmented and aligned and chunked")
    print(train_chunks_aligned)
    print(val_chunks_aligned)
    print(test_chunks_aligned)
    
    # Create filtered versions (non-aligned) with same processing as original
    my_segment = partial(segment, align=False, max_length=128, max_cer=0.8)
    
    test_chunks_filtered = (
        test.map(my_segment, batched=False, num_proc=4)
        .filter(my_filter_bad_sample)
        .map(chunkify, batched=True, num_proc=4)
    )
    val_chunks_filtered = (
        val.map(my_segment, batched=False, num_proc=4)
        .filter(my_filter_bad_sample)
        .map(chunkify, batched=True, num_proc=4)
    )
    train_chunks_filtered = (
        train.map(my_segment, batched=False, num_proc=4)
        .filter(my_filter_bad_sample)
        .map(chunkify, batched=True, num_proc=4)
    )
    
    print("Segmented and not aligned but filtered and chunked")
    print(train_chunks_filtered)
    print(val_chunks_filtered)
    print(test_chunks_filtered)
    
    # Calculate and print error metrics for filtered datasets
    print(
        "eval score",
        jiwer.cer(val_chunks_filtered["gt"], val_chunks_filtered["ocr"]),
        jiwer.wer(val_chunks_filtered["gt"], val_chunks_filtered["ocr"]),
    )
    print(
        "test score",
        jiwer.cer(test_chunks_filtered["gt"], test_chunks_filtered["ocr"]),
        jiwer.wer(test_chunks_filtered["gt"], test_chunks_filtered["ocr"]),
    )
    
    return (
        train_chunks_filtered,
        val_chunks_filtered,
        test_chunks_filtered,
        train_chunks_aligned,
        val_chunks_aligned,
        test_chunks_aligned,
    )

def prepare_ocr_datasets(input_file, ocr_systems, args, test_fns=None, val_fns=None, seed=666):
    """Process the input file to extract data for each OCR system"""
    # Dictionary to store prepared datasets for each OCR system
    ocr_datasets = {}
    
    # Load the input JSON file
    with open(input_file, "r", encoding="utf-8") as file:
        data = []
        for line in file:
            data.append(json.loads(line))
    
    # Process each OCR system
    for system_name, system_key in ocr_systems.items():
        print(f"\nProcessing OCR system: {system_name} (key: {system_key})")
        
        # Extract data for this OCR system
        system_data = []
        for item in data:
            transcription = item.get("transcription", "").lstrip("\ufeff").strip()  # Ground truth
            ocr_text = item.get("ocr", {}).get(system_key, "").strip()  # OCR output
            
            # Only add if both transcription and OCR text exist
            if transcription and ocr_text:
                system_data.append({
                    "file": item.get("id", ""),
                    "gt": transcription,
                    "ocr": ocr_text
                })
        
        print(f"Extracted {len(system_data)} samples for {system_name}")
        
        # Save to temporary file for processing
        temp_file = f"data/thesis_ocr_output/temp_{system_name}.json"
        with open(temp_file, "w", encoding="utf-8") as out_file:
            json.dump(system_data, out_file, ensure_ascii=False)
        
        # Process the data using the existing function
        system_dataset = prepare_data_from_json(temp_file, args, test_fns, val_fns, seed)
        ocr_datasets[system_name] = system_dataset
        
        # Clean up temporary file
        os.remove(temp_file)
    
    return ocr_datasets

if __name__ == "__main__":
    seed = 666
    
    training_args, args = get_args()
    
    # Define OCR systems to process
    ocr_systems = {
        "tesseract": "tesseract_v5.5.0_new_swebest_oem1_psm6",
        "doctr": "doctr", 
        "idefics": "idefics",
        "kraken": "kraken-german"
    }
    
    # Input file with multiple OCR outputs
    input_file = "data/thesis_ocr_output/processed_combined_json.json"
    
    # No test files for now
    test_fns = None
    val_fns = None
    
    # Process all OCR systems
    ocr_datasets = prepare_ocr_datasets(input_file, ocr_systems, args, test_fns, val_fns, seed)
    
    # Define the train-test combinations based on the provided table
    train_test_pairs = [
        # Baselines (same system for train and test)
        ("tesseract", "tesseract", "baseline_1"),
        ("kraken", "kraken", "baseline_2"),
        ("doctr", "doctr", "baseline_3"),
        ("idefics", "idefics", "baseline_4"),
        
        # Cross-system evaluations
        ("tesseract", "doctr", "result_1a"),
        ("tesseract", "kraken", "result_1b"),
        ("tesseract", "idefics", "result_1c"),
        
        ("kraken", "tesseract", "result_2a"),
        ("kraken", "doctr", "result_2b"),
        ("kraken", "idefics", "result_2c"),
        
        ("doctr", "tesseract", "result_3a"),
        ("doctr", "kraken", "result_3b"),
        ("doctr", "idefics", "result_3c"),
        
        ("idefics", "tesseract", "result_4a"),
        ("idefics", "doctr", "result_4b"),
        ("idefics", "kraken", "result_4c")
    ]
    
    # Results dictionary to store all evaluations
    results = []
    
    # Process each train-test pair
    for train_system, test_system, experiment_name in train_test_pairs:
        print(f"\n{'='*80}")
        print(f"Processing experiment: {experiment_name}")
        print(f"Training on {train_system}, Testing on {test_system}")
        print(f"{'='*80}\n")
        
        # Get the datasets
        if train_system not in ocr_datasets or test_system not in ocr_datasets:
            print(f"Warning: Missing data for {train_system} or {test_system}, skipping...")
            continue
        
        # Get train and test datasets
        train_data = ocr_datasets[train_system]
        test_data = ocr_datasets[test_system]
        
        # Use appropriate datasets for training and testing
        train_dataset = train_data[0]  # train_chunks_filtered
        val_dataset = train_data[1]    # val_chunks_filtered
        
        # For testing, use the test data from the test OCR system
        if train_system == test_system:
            # For baseline, use the same OCR system's test set
            test_dataset = test_data[2]  # test_chunks_filtered
        else:
            # For cross-system, use the test OCR system's test set
            test_dataset = test_data[2]  # test_chunks_filtered
        
        # Print dataset sizes
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Calculate baseline metrics for test data
        baseline_cer = jiwer.cer(test_dataset["gt"], test_dataset["ocr"])
        baseline_wer = jiwer.wer(test_dataset["gt"], test_dataset["ocr"])
        
        print(f"Baseline Test CER: {baseline_cer:.4f}, WER: {baseline_wer:.4f}")
        
        # Create output directory for this experiment
        experiment_dir = os.path.join(args.output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Update training args
        training_args.output_dir = experiment_dir
        
        # Run training and evaluation
        print(f"Starting training for {experiment_name}...")
        
        try:
            # Uncomment to actually run the training
            model_results = run_training(train_dataset, val_dataset, test_dataset, training_args=training_args)
            
            # For now, just create a placeholder for results
            model_results = {
                "test_cer": 0.0,  # This would come from actual training
                "test_wer": 0.0   # This would come from actual training
            }
            
            # Store results
            result = {
                "name": experiment_name,
                "train_system": train_system,
                "test_system": test_system,
                "baseline_cer": baseline_cer,
                "baseline_wer": baseline_wer,
                "model_cer": model_results.get("test_cer", 0.0),
                "model_wer": model_results.get("test_wer", 0.0),
                "improvement_cer": baseline_cer - model_results.get("test_cer", 0.0),
                "improvement_wer": baseline_wer - model_results.get("test_wer", 0.0)
            }
            
            results.append(result)
            
            # Save results after each experiment
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
                
            print(f"Completed experiment: {experiment_name}")
            
        except Exception as e:
            print(f"Error running experiment {experiment_name}: {e}")
    
    # Print final summary
    print("\nExperiment Summary:")
    print("=" * 80)
    for result in results:
        print(f"{result['name']}: Train={result['train_system']}, Test={result['test_system']}")
        print(f"  Baseline CER: {result['baseline_cer']:.4f}, WER: {result['baseline_wer']:.4f}")
        print(f"  Model CER: {result['model_cer']:.4f}, WER: {result['model_wer']:.4f}")
        print(f"  Improvement: CER: {result['improvement_cer']:.4f}, WER: {result['improvement_wer']:.4f}")
        print("-" * 60)