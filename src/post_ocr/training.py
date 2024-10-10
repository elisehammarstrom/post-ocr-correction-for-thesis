import logging
import os
from functools import partial


import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

from post_ocr.arguments import get_args
from post_ocr.data import prepare_for_training
from post_ocr.utils import compute_metrics, setup_logger, batch_decode


logger = logging.getLogger(__name__)


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64, 128]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1, log=True),
        "optim": trial.suggest_categorical("optim", ["adafactor", "adamw_torch"]),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type", ["cosine", "linear", "constant_with_warmup"]
        ),
    }


def compute_objective(metrics: dict[str, float]) -> list[float]:
    return [metrics["eval_wer"], metrics["eval_cer"]]


def run_training(train_dataset, eval_dataset, test_dataset):
    training_args, args = get_args()

    logger = setup_logger(args)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Data collator
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if args.fp16 else None,
        )

    my_compute_metrics = partial(compute_metrics, tokenizer=tokenizer)

    my_prepare = partial(prepare_for_training, tokenizer=tokenizer, args=args)

    remove_columns = ["ocr", "gt", "file"]
    train_dataset = train_dataset.map(
        my_prepare,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=remove_columns,
        desc="Running tokenizer on train data",
    )
    eval_dataset = eval_dataset.map(
        my_prepare,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=remove_columns,
        desc="Running tokenizer on eval data",
    )
    test_dataset = test_dataset.map(
        my_prepare,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=remove_columns,
        desc="Running tokenizer on eval data",
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=my_compute_metrics if training_args.predict_with_generate else None,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Training
    if args.do_train:
        logger.info("*** Training ***")
        if args.hypertune:

            def model_init(trial):
                return AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

            # Initialize our Trainer
            trainer = Seq2SeqTrainer(
                model=None,
                args=training_args,
                train_dataset=train_dataset if args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=my_compute_metrics
                if training_args.predict_with_generate
                else None,
                model_init=model_init,
            )

            best_trial = trainer.hyperparameter_search(
                direction=["minimize", "minimize"],
                backend="optuna",
                hp_space=optuna_hp_space,
                n_trials=20,
                compute_objective=compute_objective,
            )
            logger.info("Finished Hyperparamter search")
            logger.info(best_trial)
            trainer.apply_hyperparameters(best_trial.hyperparameters, final_model=True)
            train_result = trainer.train()
        else:
            checkpoint = None
            if args.resume_from_checkpoint is not None:
                checkpoint = args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else args.val_max_target_length
    )
    num_beams = (
        args.num_beams if args.num_beams is not None else training_args.generation_num_beams
    )

    logger.info("Model Generation Config")
    logger.info(model.generation_config)
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=max_length,
            num_beams=num_beams,
            metric_key_prefix="eval",
        )
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset,
            metric_key_prefix="predict",
            max_length=max_length,
            num_beams=num_beams,
        )
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(test_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = batch_decode(predictions, tokenizer)
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt"
                )
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))
