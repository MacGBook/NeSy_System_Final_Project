# Predicate Trainer

Train two DistilBERT binary classifiers that return scores for:

- `has_sarcasm`
- `is_questioning`

## Datasets

- Sarcasm: Hugging Face `tweet_eval` with the `irony` configuration. The positive class is treated as `has_sarcasm`.
- Questioning: positive examples from the `question` field in Hugging Face `squad`, paired with balanced negative statement examples from `ag_news`.

The datasets and the `distilbert-base-uncased` checkpoint are downloaded by the Hugging Face libraries the first time you train.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

Train both models:

```powershell
python train_predicates.py --task all
```

Train only one model:

```powershell
python train_predicates.py --task sarcasm
python train_predicates.py --task questioning
```

Quick smoke run:

```powershell
python train_predicates.py --task all --max-train-samples 200 --epochs 1
```

Outputs are saved to:

- `models/sarcasm`
- `models/questioning`

Intermediate Trainer checkpoints are cleaned up by default to avoid storing duplicate optimizer/model state. Add `--keep-checkpoints` if you want resumable checkpoints.

## Score Text

```powershell
python score_predicates.py "Oh great, another meeting." "Are you coming today?"
```

JSON lines output:

```powershell
python score_predicates.py --json "Oh great, another meeting." "Are you coming today?"
```

Example result shape:

```json
{"text":"Are you coming today?","has_sarcasm":0.0312,"is_questioning":0.9844}
```
