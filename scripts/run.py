from functools import partial

import jiwer
import json

from post_ocr.data import chunkify, filter_bad_sample, segment
from post_ocr.arguments import get_args
from post_ocr.training import run_training

from datasets import Dataset, concatenate_datasets

def load_data_from_jsonl(json_path, ocr_key):
    """Load data from a JSONL file for a specific OCR system"""
    data = []
    with open(json_path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            transcription = item.get("transcription", "").lstrip("\ufeff").strip()
            ocr_text = item.get("ocr", {}).get(ocr_key, "").strip()
            
            if transcription and ocr_text:
                data.append({
                    "file": item.get("id", ""),
                    "gt": transcription,
                    "ocr": ocr_text
                })
    
    return Dataset.from_dict({
        "file": [item["file"] for item in data],
        "gt": [item["gt"] for item in data],
        "ocr": [item["ocr"] for item in data]
    })

def prepare_data(ocr_key, json_path, args, test_fns=None, val_fns=None):
    """Prepare data from JSONL input instead of kubhist dataset"""
    split_size = 0.15

    ds = load_data_from_jsonl(json_path, ocr_key)
    
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

    my_filter_bad_sample = partial(filter_bad_sample, min_len=4, max_cer=0.5)

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

    # print("test score")
    # print(jiwer.cer(test_chunks_aligned["gt"], test_chunks_aligned["ocr"]))

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

    json_path = "data/thesis_ocr_output/processed_combined_json.json"

    test_fns = None
    val_fns = None

    tesseract = prepare_data("tesseract_v5.5.0_new_swebest_oem1_psm6", json_path, args, test_fns=test_fns)

    # Use the same test and validation files for consistency
    test_fns = [x.split("/")[-1] for x in tesseract[2]["file"]]
    val_fns = [x.split("/")[-1] for x in tesseract[1]["file"]]

    # Split the other ocr systems 
    doctr = prepare_data("doctr", json_path, args, test_fns, val_fns)
    kraken = prepare_data("kraken-german", json_path, args, test_fns, val_fns)
    idefics = prepare_data("idefics", json_path, args, test_fns, val_fns)

    # Create datasets for baseline evals
    tesseract_train = concatenate_datasets([tesseract[0]])
    tesseract_val = concatenate_datasets([tesseract[1]])
    tesseract_test = concatenate_datasets([tesseract[2]])

    doctr_train = concatenate_datasets([doctr[0]])
    doctr_val = concatenate_datasets([doctr[1]])
    doctr_test = concatenate_datasets([doctr[2]])

    kraken_train = concatenate_datasets([kraken[0]])
    kraken_val = concatenate_datasets([kraken[1]])
    kraken_test = concatenate_datasets([kraken[2]])

    idefics_train = concatenate_datasets([idefics[0]])
    idefics_val = concatenate_datasets([idefics[1]])
    idefics_test = concatenate_datasets([idefics[2]])

    # Create aligned versions
    _tesseract_train = concatenate_datasets([tesseract[3]])
    _tesseract_val = concatenate_datasets([tesseract[4]])
    _tesseract_test = concatenate_datasets([tesseract[5]])

    _doctr_train = concatenate_datasets([doctr[3]])
    _doctr_val = concatenate_datasets([doctr[4]])
    _doctr_test = concatenate_datasets([doctr[5]])

    _kraken_train = concatenate_datasets([kraken[3]])
    _kraken_val = concatenate_datasets([kraken[4]])
    _kraken_test = concatenate_datasets([kraken[5]])

    _idefics_train = concatenate_datasets([idefics[3]])
    _idefics_val = concatenate_datasets([idefics[4]])
    _idefics_test = concatenate_datasets([idefics[5]])

    print("---------- Tesseract ----------")
    print(
        "eval score",
        jiwer.cer(tesseract_val["gt"], tesseract_val["ocr"]),
        jiwer.wer(tesseract_val["gt"], tesseract_val["ocr"]),
    )
    print(
        "eval score aligned",
        jiwer.cer(_tesseract_val["gt"], _tesseract_val["ocr"]),
        jiwer.wer(_tesseract_val["gt"], _tesseract_val["ocr"]),
    )
    print(
        "test score",
        jiwer.cer(tesseract_test["gt"], tesseract_test["ocr"]),
        jiwer.wer(tesseract_test["gt"], tesseract_test["ocr"]),
    )
    print(
        "test score aligned",
        jiwer.cer(_tesseract_test["gt"], _tesseract_test["ocr"]),
        jiwer.wer(_tesseract_test["gt"], _tesseract_test["ocr"]),
    )

    print("---------- DocTR ----------")
    print(
        "eval score",
        jiwer.cer(doctr_val["gt"], doctr_val["ocr"]),
        jiwer.wer(doctr_val["gt"], doctr_val["ocr"]),
    )
    print(
        "eval score aligned",
        jiwer.cer(_doctr_val["gt"], _doctr_val["ocr"]),
        jiwer.wer(_doctr_val["gt"], _doctr_val["ocr"]),
    )
    print(
        "test score",
        jiwer.cer(doctr_test["gt"], doctr_test["ocr"]),
        jiwer.wer(doctr_test["gt"], doctr_test["ocr"]),
    )
    print(
        "test score aligned",
        jiwer.cer(_doctr_test["gt"], _doctr_test["ocr"]),
        jiwer.wer(_doctr_test["gt"], _doctr_test["ocr"]),
    )

    print("---------- Kraken ----------")
    print(
        "eval score",
        jiwer.cer(kraken_val["gt"], kraken_val["ocr"]),
        jiwer.wer(kraken_val["gt"], kraken_val["ocr"]),
    )
    print(
        "eval score aligned",
        jiwer.cer(_kraken_val["gt"], _kraken_val["ocr"]),
        jiwer.wer(_kraken_val["gt"], _kraken_val["ocr"]),
    )
    print(
        "test score",
        jiwer.cer(kraken_test["gt"], kraken_test["ocr"]),
        jiwer.wer(kraken_test["gt"], kraken_test["ocr"]),
    )
    print(
        "test score aligned",
        jiwer.cer(_kraken_test["gt"], _kraken_test["ocr"]),
        jiwer.wer(_kraken_test["gt"], _kraken_test["ocr"]),
    )

    print("---------- Idefics ----------")
    print(
        "eval score",
        jiwer.cer(idefics_val["gt"], idefics_val["ocr"]),
        jiwer.wer(idefics_val["gt"], idefics_val["ocr"]),
    )
    print(
        "eval score aligned",
        jiwer.cer(_idefics_val["gt"], _idefics_val["ocr"]),
        jiwer.wer(_idefics_val["gt"], _idefics_val["ocr"]),
    )
    print(
        "test score",
        jiwer.cer(idefics_test["gt"], idefics_test["ocr"]),
        jiwer.wer(idefics_test["gt"], idefics_test["ocr"]),
    )
    print(
        "test score aligned",
        jiwer.cer(_idefics_test["gt"], _idefics_test["ocr"]),
        jiwer.wer(_idefics_test["gt"], _idefics_test["ocr"]),
    )

    # print("\n===== BASELINE EXPERIMENTS =====") # Uncomment me!
    
    # print("\n=== Tesseract baseline ===")
    # run_training(tesseract_train, tesseract_val, tesseract_test)
    
    # print("\n=== Kraken baseline ===")
    # run_training(kraken_train, kraken_val, kraken_test)
    
    # print("\n=== DocTR baseline ===")
    # run_training(doctr_train, doctr_val, doctr_test)
    
    # print("\n=== IDEFICS baseline ===")
    # run_training(idefics_train, idefics_val, idefics_test)
    
    # # Cross-system evaluations
    # print("\n===== CROSS-SYSTEM EXPERIMENTS =====")
    
    # # Tesseract as training data
    # print("\n=== Result 1a: Tesseract → DocTR ===")
    # run_training(tesseract_train, tesseract_val, doctr_test)
    
    # print("\n=== Result 1b: Tesseract → Kraken ===")
    # run_training(tesseract_train, tesseract_val, kraken_test)
    
    # print("\n=== Result 1c: Tesseract → IDEFICS ===")
    # run_training(tesseract_train, tesseract_val, idefics_test)
    
    # # Kraken as training data
    # print("\n=== Result 2a: Kraken → Tesseract ===")
    # run_training(kraken_train, kraken_val, tesseract_test)
    
    # print("\n=== Result 2b: Kraken → DocTR ===")
    # run_training(kraken_train, kraken_val, doctr_test)
    
    # print("\n=== Result 2c: Kraken → IDEFICS ===")
    # run_training(kraken_train, kraken_val, idefics_test)
    
    # # DocTR as training data
    # print("\n=== Result 3a: DocTR → Tesseract ===")
    # run_training(doctr_train, doctr_val, tesseract_test)
    
    # print("\n=== Result 3b: DocTR → Kraken ===")
    # run_training(doctr_train, doctr_val, kraken_test)
    
    # print("\n=== Result 3c: DocTR → IDEFICS ===")
    # run_training(doctr_train, doctr_val, idefics_test)
    
    # # IDEFICS as training data
    # print("\n=== Result 4a: IDEFICS → Tesseract ===")
    # run_training(idefics_train, idefics_val, tesseract_test)
    
    # print("\n=== Result 4b: IDEFICS → DocTR ===")
    # run_training(idefics_train, idefics_val, doctr_test)
    
    # print("\n=== Result 4c: IDEFICS → Kraken ===")
    # run_training(idefics_train, idefics_val, kraken_test)

