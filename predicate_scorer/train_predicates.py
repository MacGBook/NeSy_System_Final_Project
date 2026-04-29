import argparse
import inspect
import random
import shutil
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


TASKS = ("sarcasm", "questioning")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_sarcasm_dataset() -> DatasetDict:
    dataset = load_dataset("tweet_eval", "irony")

    def normalize(example):
        return {"text": example["text"], "labels": int(example["label"])}

    return DatasetDict(
        {
            split: dataset[split].map(normalize, remove_columns=dataset[split].column_names)
            for split in ("train", "validation", "test")
        }
    )


def _balanced_binary_dataset(
    positive_texts,
    negative_texts,
    seed: int,
    max_positive: int | None = None,
) -> Dataset:
    positive_texts = list(positive_texts)
    negative_texts = list(negative_texts)
    limit = min(len(positive_texts), len(negative_texts))
    if max_positive is not None:
        limit = min(limit, max_positive)

    positive_texts = positive_texts[:limit]
    negative_texts = negative_texts[:limit]
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    return Dataset.from_dict({"text": texts, "labels": labels}).shuffle(seed=seed)


def load_questioning_dataset(seed: int) -> DatasetDict:
    questions = load_dataset("squad")
    statements = load_dataset("ag_news")

    statement_train = statements["train"].shuffle(seed=seed)
    statement_test = statements["test"].shuffle(seed=seed)

    train_full = _balanced_binary_dataset(
        positive_texts=questions["train"]["question"],
        negative_texts=statement_train["text"],
        seed=seed,
    )
    train_validation = train_full.train_test_split(test_size=0.1, seed=seed)

    test = _balanced_binary_dataset(
        positive_texts=questions["validation"]["question"],
        negative_texts=statement_test["text"],
        seed=seed,
    )

    return DatasetDict(
        {
            "train": train_validation["train"],
            "validation": train_validation["test"],
            "test": test,
        }
    )


def maybe_limit_dataset(dataset: DatasetDict, max_train_samples: int | None) -> DatasetDict:
    if max_train_samples is None:
        return dataset

    limited = DatasetDict()
    for split, split_dataset in dataset.items():
        limit = min(max_train_samples, len(split_dataset))
        if split != "train":
            limit = min(max(1, max_train_samples // 5), len(split_dataset))
        limited[split] = split_dataset.shuffle(seed=13).select(range(limit))
    return limited


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True)


def make_training_arguments(args, output_dir: Path) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "logging_steps": args.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "save_only_model" in signature.parameters:
        kwargs["save_only_model"] = True

    return TrainingArguments(**kwargs)


def cleanup_checkpoints(output_dir: Path) -> None:
    for checkpoint in output_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)


def train_task(task: str, args) -> None:
    output_dir = Path(args.output_dir) / task
    label_names = {
        "sarcasm": {0: "not_sarcasm", 1: "has_sarcasm"},
        "questioning": {0: "not_questioning", 1: "is_questioning"},
    }[task]

    print(f"\n=== Training {task} classifier ===")
    if task == "sarcasm":
        raw_dataset = load_sarcasm_dataset()
    else:
        raw_dataset = load_questioning_dataset(seed=args.seed)

    raw_dataset = maybe_limit_dataset(raw_dataset, args.max_train_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = tokenize_dataset(raw_dataset, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=label_names,
        label2id={label: idx for idx, label in label_names.items()},
    )

    trainer_kwargs = {
        "model": model,
        "args": make_training_arguments(args, output_dir),
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(f"Test metrics for {task}: {metrics}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if not args.keep_checkpoints:
        cleanup_checkpoints(output_dir)
    print(f"Saved {task} model to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DistilBERT binary predicate classifiers for sarcasm and questioning."
    )
    parser.add_argument(
        "--task",
        choices=("all", *TASKS),
        default="all",
        help="Which classifier to train.",
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Base Hugging Face model checkpoint.",
    )
    parser.add_argument("--output-dir", default="models", help="Where trained models are saved.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional quick-smoke limit per training split.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep intermediate Trainer checkpoints for resuming training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        train_task(task, args)


if __name__ == "__main__":
    main()
