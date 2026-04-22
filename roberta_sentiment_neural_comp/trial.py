# IMPORTS

# Referenced following links: https://medium.com/@s.sadathosseini/sentiment-analysis-on-twitter-data-using-roberta-model-in-google-colab-b7bb5a9b03fc
# training dataset: https://www.kaggle.com/datasets/imaadmahmood/social-media-and-misinformation-dataset-2024/data 

import os
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import csv

from transformers import DataCollatorWithPadding


import torch
import numpy as np
from scipy.special import softmax


#################################################################
# LOAD TRAINING DATA

texts = []
labels = []

with open("social media content and misinformation data.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        texts.append(row["Content_Text"])
        labels.append(float(row["Sentiment_Score"]))

#################################################################
# CONVERT TO CLASSES FOR TRAINING

final_labels = []

for score in labels:
    if score < 0:
        final_labels.append(0)   # Negative
    elif score == 0:
        final_labels.append(1)   # Neutral
    else:
        final_labels.append(2)   # Positive

#################################################################
# CREATE DATASET FOR TRAINING

dataset = Dataset.from_dict({
    "text": texts,
    "label": final_labels
})

#################################################################
# TOKENIZE

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128   
    )

dataset = dataset.map(tokenize, batched=True)

#################################################################
# TRAIN/TEST SPLIT

dataset = dataset.train_test_split(test_size=0.2)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

#################################################################
# LOAD MODEL

model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment",
    num_labels=3
)

#################################################################
# TRAINING SETUP

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator   
)

#################################################################
# TRAINING THE MODEL

trainer.train()

#################################################################
# SAVEING THE MODEL FOR FUTURE USE

trainer.save_model("my_sentiment_model")
tokenizer.save_pretrained("my_sentiment_model")

#################################################################
#################################################################
#################################################################
# Getting Sentiment Output From Tweet Text From Our Trained Model

model_path = "my_sentiment_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

labels = ["Negative", "Neutral", "Positive"]

def predict_sentiment(text):

    # tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # run model (no gradients needed)
    with torch.no_grad():
        outputs = model(**inputs)

    # convert logits → probabilities
    scores = outputs.logits[0].numpy()
    scores = softmax(scores)

    # get best label
    predicted_class = np.argmax(scores)

    return {
        "text": text,
        "label": labels[predicted_class],
        "confidence": float(scores[predicted_class]),
        "all_scores": {
            labels[i]: float(scores[i]) for i in range(len(labels))
        }
    }

#################################################################
# USING THE MODEL FOR PREDICTION- ABLATION 1


list_of_test_tweets = []

with open("flat_earth_tweets.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        list_of_test_tweets.append(row["text"])


neg_total_results = []
neut_total_results = []
pos_total_results = []

for tweet in list_of_test_tweets:
    result = predict_sentiment(tweet)
    print(result)
    neg_total_results.append(result['all_scores']['Negative'])
    neut_total_results.append(result['all_scores']['Neutral'])
    pos_total_results.append(result['all_scores']['Positive'])

with open("twitter_sentiment_output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    
    writer.writerow(["Text", "Negative", "Neutral", "Positive"])
    
    for ind in range(len(neg_total_results)):
        text_a1_item = list_of_test_tweets[ind] 
        negitem = neg_total_results[ind]
        neutitem = neut_total_results[ind]
        positem = pos_total_results[ind]

        writer.writerow([text_a1_item, negitem, neutitem, positem])

#################################################################
# USING THE MODEL FOR PREDICTION- ABLATION 2

self_text = []
title = []

with open("reddit_flat_earth_source_posts_clean.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        title.append(row["title"])
        self_text.append(row["selftext"])

total_test = []

for str_ind in range(len(self_text)):
    title_str = str(title[str_ind])
    self_text_str = str(self_text[str_ind])
    combined_string = title_str + self_text_str
    total_test.append(combined_string)

a2_neg_total_results = []
a2_neut_total_results = []
a2_pos_total_results = []

for tweet in total_test:
    result = predict_sentiment(tweet)
    print(result)
    a2_neg_total_results.append(result['all_scores']['Negative'])
    a2_neut_total_results.append(result['all_scores']['Neutral'])
    a2_pos_total_results.append(result['all_scores']['Positive'])

with open("reddit_sentiment_output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    
    writer.writerow(["Text", "Negative", "Neutral", "Positive"])
    
    for ind in range(len(a2_neg_total_results)):
        testitem = total_test[ind]
        negitem = a2_neg_total_results[ind]
        neutitem = a2_neut_total_results[ind]
        positem = a2_pos_total_results[ind]

        writer.writerow([testitem, negitem, neutitem, positem])