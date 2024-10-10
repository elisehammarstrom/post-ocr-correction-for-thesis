import logging
import multiprocessing as mp
from functools import partial

import jiwer
import numpy as np


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    #decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)
    decoded_preds = batch_decode(preds, tokenizer)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)
    decoded_labels = batch_decode(labels, tokenizer)

    # Some simple post-processing
    # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = {
        "cer": jiwer.cer(decoded_labels, decoded_preds),
        "wer": jiwer.wer(decoded_labels, decoded_preds),
    }

    return result


def setup_logger(args):
    numeric_level = getattr(logging, args.log_level_py.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.log_level)
    logging.basicConfig(level=numeric_level, filename=args.log_file, encoding="utf-8")
    logger = logging.getLogger(__name__)
    return logger


def decode(x, tokenizer):
    try:
        return tokenizer.convert_tokens_to_string(
            "".join([tokenizer._convert_id_to_token(i) for i in x if i > 2 and i < 256 + 3])
        )
    except Exception as e:
        print(e)
        print(list(x))
        print("".join([tokenizer._convert_id_to_token(i) for i in x if i > 2]))
        raise Exception(e)


def batch_decode(input_ids, tokenizer):
    my_decode = partial(decode, tokenizer=tokenizer)
    with mp.Pool(processes=16) as pool:
        results = []
        for r in pool.imap(my_decode, input_ids, chunksize=100):
            results.append(r)
    return results
