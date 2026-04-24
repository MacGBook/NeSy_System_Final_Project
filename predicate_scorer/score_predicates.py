import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


POSITIVE_LABELS = {
    "has_sarcasm",
    "LABEL_1",
    "is_questioning",
}


def positive_label_id(model) -> int:
    for label, idx in model.config.label2id.items():
        if label in POSITIVE_LABELS:
            return int(idx)
    return 1


def load_model(model_dir: str):
    path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model, positive_label_id(model)


@torch.no_grad()
def score_texts(texts: list[str], model_dir: str) -> list[float]:
    tokenizer, model, positive_id = load_model(model_dir)
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities[:, positive_id].cpu().tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="Score text with trained predicate classifiers.")
    parser.add_argument("text", nargs="+", help="One or more texts to score.")
    parser.add_argument("--sarcasm-model", default="models/sarcasm")
    parser.add_argument("--questioning-model", default="models/questioning")
    parser.add_argument("--json", action="store_true", help="Print JSON lines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sarcasm_scores = score_texts(args.text, args.sarcasm_model)
    questioning_scores = score_texts(args.text, args.questioning_model)

    rows = [
        {
            "text": text,
            "has_sarcasm": sarcasm,
            "is_questioning": questioning,
        }
        for text, sarcasm, questioning in zip(args.text, sarcasm_scores, questioning_scores)
    ]

    if args.json:
        for row in rows:
            print(json.dumps(row))
        return

    for row in rows:
        print(f"text: {row['text']}")
        print(f"  has_sarcasm:   {row['has_sarcasm']:.4f}")
        print(f"  is_questioning: {row['is_questioning']:.4f}")


if __name__ == "__main__":
    main()
