from functools import partial
import jiwer
import json

from post_ocr.data import chunkify, filter_bad_sample, segment
from post_ocr.arguments import get_args
from post_ocr.training import run_training
from datasets import Dataset, DatasetDict, concatenate_datasets

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

if __name__ == "__main__":
    seed = 666
    
    training_args, args = get_args()
    
    # Load test file names if available (similar to original)
    try:
        test_fns = json.load(open("data/thesis_ocr_output/holdout_files.json"))
    except FileNotFoundError:
        test_fns = None
        val_fns = None
    
    # Prepare data from your JSON file
    main_dataset = prepare_data_from_json(
        "data/thesis_ocr_output/prepared_data.json", 
        args, 
        test_fns=test_fns
    )
    
    # If you have multiple OCR outputs, you could load them separately and concatenate
    # Otherwise, just use the main dataset for all parts to maintain structure
    train_dataset = main_dataset[0]
    val_dataset = main_dataset[1]
    test_dataset = main_dataset[2]
    
    _train_dataset = main_dataset[3]
    _val_dataset = main_dataset[4]
    _test_dataset = main_dataset[5]
    
    # Print metrics like the original
    print(
        "eval score",
        jiwer.cer(val_dataset["gt"], val_dataset["ocr"]),
        jiwer.wer(val_dataset["gt"], val_dataset["ocr"]),
    )
    print(
        "eval score aligned",
        jiwer.cer(_val_dataset["gt"], _val_dataset["ocr"]),
        jiwer.wer(_val_dataset["gt"], _val_dataset["ocr"]),
    )
    print(
        "test score",
        jiwer.cer(test_dataset["gt"], test_dataset["ocr"]),
        jiwer.wer(test_dataset["gt"], test_dataset["ocr"]),
    )
    print(
        "test score aligned",
        jiwer.cer(_test_dataset["gt"], _test_dataset["ocr"]),
        jiwer.wer(_test_dataset["gt"], _test_dataset["ocr"]),
    )
    
    # Run the training with the prepared datasets
    run_training(train_dataset, val_dataset, test_dataset)