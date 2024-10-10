from functools import partial

import jiwer
import json

from post_ocr.data import chunkify, filter_bad_sample, load_data, segment
from post_ocr.arguments import get_args
from post_ocr.training import run_training


from datasets import concatenate_datasets


def prepare_data(gt, ocr, args, test_fns=None, val_fns=None):

    split_size = 0.15

    kh_root = "data/kubhist/kubhist/"
    ds = load_data(
        f"{kh_root}/gt/{gt}/",
        f"{kh_root}/ocr/{ocr}/",
        cache_dir=args.cache_dir,
    )
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
        test.map(my_segment).filter(my_filter_bad_sample).map(chunkify, batched=True, num_proc=4)
    )
    val_chunks_filtered = (
        val.map(my_segment).filter(my_filter_bad_sample).map(chunkify, batched=True, num_proc=4)
    )
    train_chunks_filtered = (
        train.map(my_segment).filter(my_filter_bad_sample).map(chunkify, batched=True, num_proc=4)
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

    test_fns = json.load(open("data/kubhist/og_holdout_files.json"))

    abbyy = prepare_data("no-long-s", "Abbyy", args, test_fns=test_fns)
    # test_fns = [x.split("/")[-1] for x in abbyy[2]["file"]]
    val_fns = [x.split("/")[-1] for x in abbyy[1]["file"]]

    abbyy_tesseract = prepare_data("no-long-s", "Abbyy-Tesseract", args, test_fns, val_fns)
    tesseract = prepare_data("long-s", "Tesseract", args, test_fns, val_fns)

    train_dataset = concatenate_datasets([abbyy[0], abbyy_tesseract[0], tesseract[0]])
    val_dataset = concatenate_datasets([abbyy[1], abbyy_tesseract[1], tesseract[1]])
    test_dataset = concatenate_datasets([abbyy[2], abbyy_tesseract[2], tesseract[2]])

    _train_dataset = concatenate_datasets([abbyy[3], abbyy_tesseract[3], tesseract[3]])
    _val_dataset = concatenate_datasets([abbyy[4], abbyy_tesseract[4], tesseract[4]])
    _test_dataset = concatenate_datasets([abbyy[5], abbyy_tesseract[5], tesseract[5]])


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
    # print(model_args)
    # print(data_args)
    # print(training_args)
    #
    # print(model_args)
    # print(model_args.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(
    #    model_args.model_name_or_path, cache_dir=model_args.cache_dir
    # )

    # import time

    # print("encode")
    # start = time.time()
    # ocr_test_tok = tokenizer(test_chunks_filtered["ocr"])
    # print(f"encoding took {time.time() - start:.4f}s")

    ## print("decode")
    ## start = time.time()
    ## ocr_test_detok = batch_decode(ocr_test_tok["input_ids"], tokenizer)
    ## print(f"decoding took {time.time() - start:.4f}s")

    # from transformers.trainer_utils import EvalPrediction

    # print(tokenizer.pad_token_id)
    # print(
    #    compute_metrics(
    #        EvalPrediction(
    #            tokenizer(test_chunks_filtered["ocr"])["input_ids"],
    #            tokenizer(test_chunks_filtered["gt"])["input_ids"],
    #        ),
    #        tokenizer,
    #    )
    # )

    run_training(train_dataset, val_dataset, test_dataset)
