import re
import os

import datasets
import jiwer
from tqdm import tqdm

import post_ocr.diff as diff


def load_data(gt_dir, ocr_dir, cache_dir):
    ocr_files = [
        os.path.join(parent, file)
        for parent, dirs, files in os.walk(ocr_dir)
        if not dirs
        for file in files
    ]

    def _dataset_generator():
        for ocr_file in tqdm(ocr_files, desc=f"Loading {ocr_dir} and {gt_dir}"):
            gt_file = ocr_file.replace(ocr_dir, gt_dir)
            if not os.path.exists(gt_file):
                print(ocr_file, gt_file)
                continue

            # Read OCR file
            with open(ocr_file, "r") as f:
                ocr = f.read()

            # Read GT file
            with open(gt_file, "r") as f:
                gt = f.read()
            yield {"ocr": ocr, "gt": gt, "file": ocr_file.split("ocr")[-1].strip("/")}

    return datasets.Dataset.from_generator(_dataset_generator, cache_dir=cache_dir)


def align_lines(gt_lines, ocr_lines, max_cer=0.8, limit_length=40, keep_disagreements=True):
    """
    Align a and b such that the line numbers match each other.

    Arguments:
        gt_lines, ocr_lines: lists of text lines (strings)
        max_cer: How much two lines can differ and still be considered equal
        limit_length: Compare strings up to limit_length characters
        keep_disagreements: Add empty lines to align texts whenever a line
            only exists in one of the documents. If False, remove all lines
            where disagreement is over max_cer.
    """

    # Remove newlines if present
    gt_lines = [line.strip() for line in gt_lines]
    ocr_lines = [line.strip() for line in ocr_lines]

    # Shorten all lines. To speed up computation
    gt_short = [line[:limit_length] for line in gt_lines]
    ocr_short = [line[:limit_length] for line in ocr_lines]

    myers_diff = diff.myers_diff(gt_short, ocr_short, max_cer)
    new_ocr = []
    new_gt = []

    ocr_i = 0
    gt_i = 0
    for action, _ in myers_diff:
        # Action 'insert' means that 'line' is present in ocr_lines but not gt_lines.
        # Append an empty line to gt_lines to compensate.
        if action == diff.INSERT:
            if keep_disagreements:
                new_gt.append("")
                new_ocr.append(ocr_lines[ocr_i])
            ocr_i += 1

        # Action 'remove' means that 'line' is present in gt_lines but not ocr_lines.
        # Append an empty line to ocr_lines to compensate.
        elif action == diff.REMOVE:
            if keep_disagreements:
                new_ocr.append("")
                new_gt.append(gt_lines[gt_i])
            gt_i += 1

        # Action 'keep' means that 'line', present in gt_lines, has a close match in ocr_lines.
        # Keep 'line' in gt_lines, and the corresponding ocr line in ocr_lines.
        else:
            new_ocr.append(ocr_lines[ocr_i])
            new_gt.append(gt_lines[gt_i])
            ocr_i += 1
            gt_i += 1

    # Add newlines to each line
    # new_gt = [line + '\n' for line in new_gt]
    # new_ocr = [line + '\n' for line in new_ocr]
    return new_gt, new_ocr


def segment(element, align, max_length=-1, max_cer=0.8):
    """
    Split multilines OCR/GT segments line-by-line,
    allow mulitple lines if they are shorter than maxlen.
    Align segments line-by-line if align==True.

    Example: Given a max_length of 40, the following text will be split:

    Lorem ipsum dolor sit
    amet, consectetur       (len = 38)
    ----------------------
    adipiscing elit, sed
    do eiusmod tempor       (len  = 37)
    ----------------------
    incididunt ut labore    (len = 20)
    ----------------------
    et dolore magna aliqua. (len = 23)
    """

    ocr_lines = element["ocr"].splitlines()
    gt_lines = element["gt"].splitlines()

    if align:
        gt_lines, ocr_lines = align_lines(
            gt_lines, ocr_lines, keep_disagreements=False, limit_length=-1, max_cer=max_cer
        )

    ocr_chunks = []
    gt_chunks = []
    ocr_chunk = b""
    gt_chunk = b""
    for ocr_line, gt_line in zip(ocr_lines, gt_lines):
        ocr_line = bytes(ocr_line, encoding="utf-8")
        gt_line = bytes(gt_line, encoding="utf-8")

        ocr_chunk += ocr_line + bytes("\n", encoding="utf-8")
        gt_chunk += gt_line + bytes("\n", encoding="utf-8")

        if len(ocr_chunk + ocr_line) > max_length or len(gt_chunk + gt_line) > max_length:
            ocr_chunks.append(ocr_chunk[:max_length].strip())
            gt_chunks.append(gt_chunk[:max_length].strip())
            ocr_chunk = gt_chunk = b""

    if ocr_chunk or gt_chunk:
        ocr_chunks.append(ocr_chunk[:max_length].strip())
        gt_chunks.append(gt_chunk[:max_length].strip())

    element["ocr"] = [chunk.decode(encoding="utf-8", errors="ignore") for chunk in ocr_chunks]
    element["gt"] = [chunk.decode(encoding="utf-8", errors="ignore") for chunk in gt_chunks]
    return element


def filter_bad_sample(element, min_len=4, max_cer=0.5):
    """
    Filter out "bad" samples:
        len < 4, CER > 0.5 or containing `@`
    """
    if len(element["ocr"]) < min_len:
        return False
    if len(element["gt"]) < min_len:
        return False
    if "@" in element["gt"]:
        return False
    if jiwer.cer(element["gt"], element["ocr"]) >= max_cer:
        return False
    return True


def year_from_fn(fn):
    pattern = re.compile(r"-({\d})-")
    try:
        return re.search(pattern, fn).group(1)
    except Exception as _:
        return None


def chunkify(batch):
    ocr_chunks = []
    gt_chunks = []
    file_names = []
    years = []
    for o, g, f in zip(batch["ocr"], batch["gt"], batch["file"]):
        for oo, gg in zip(o, g):
            ocr_chunks.append(oo)
            gt_chunks.append(gg)
            file_names.append(f)
            years.append(year_from_fn(f))
    return {"ocr": ocr_chunks, "gt": gt_chunks, "file": file_names}


def prepare_for_training(examples, tokenizer, args):
    inputs = tokenizer(
        examples["ocr"],
        max_length=args.max_source_length,
        padding=args.padding,
        truncation=args.truncation,
    )

    labels = tokenizer(
        text_target=examples["gt"],
        max_length=args.max_target_length,
        padding=args.padding,
        truncation=args.truncation,
    )

    if args.padding == "max_length" and args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    inputs["labels"] = labels["input_ids"]

    return inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels
