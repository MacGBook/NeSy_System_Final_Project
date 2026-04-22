from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import csv
import numpy as np

#########################################################################################################################################################################
# SET UP THE TRAIN DATASET

neg_senti = []
neut_senti = []
pos_senti = []
beleive = []
disbelieve = []
question = []
mock = []

with open("train.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    reader.fieldnames = [name.strip().strip('"').replace('\ufeff', '') for name in reader.fieldnames]

    for row in reader:
        row = {k.strip('"'): v for k, v in row.items()}

        neg_senti.append(float(row["negative"])) 
        neut_senti.append(float(row["neutral"])) 
        pos_senti.append(float(row["positive"])) 
        beleive.append(float(row["beleive"])) 
        disbelieve.append(float(row["disbeleive"])) 
        question.append(float(row["question"]))
        mock.append(float(row["mock"]))


senti_tuple_list = []
bel_tuple_list = []

for current_index in range(len(neut_senti)):
    senti_tuple_bit = [neg_senti[current_index], neut_senti[current_index], pos_senti[current_index]]
    coord_tuple_bit = [beleive[current_index], disbelieve[current_index], question[current_index], mock[current_index]]
    
    senti_tuple_list.append(senti_tuple_bit)
    bel_tuple_list.append(coord_tuple_bit)

#########################################################################################################################################################################
# SET UP THE FIRST TEST DATASET

test_tweet_neg_senti = []
test_tweet_neut_senti = []
test_tweet_pos_senti = []

with open("to_test_twitter.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    reader.fieldnames = [name.strip().strip('"').replace('\ufeff', '') for name in reader.fieldnames]

    for row in reader:
        row = {k.strip('"'): v for k, v in row.items()}

        test_tweet_neg_senti.append(float(row["Negative"]))
        test_tweet_neut_senti.append(float(row["Neutral"]))
        test_tweet_pos_senti.append(float(row["Positive"]))
        

test_senti_tuple_list = []

for current_index in range(len(test_tweet_neg_senti)):
    senti_tuple_bit = [test_tweet_neg_senti[current_index], test_tweet_neut_senti[current_index], test_tweet_pos_senti[current_index]]
    
    test_senti_tuple_list.append(senti_tuple_bit)


#########################################################################################################################################################################
# TRAIN THE MLP

regr = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
regr.fit(senti_tuple_list, bel_tuple_list)

#########################################################################################################################################################################
# RUN OUR TEST DATASETS


# RUN TEST DATASET #1

test_run_1_twitter = []

for little_list in test_senti_tuple_list:
    result = regr.predict([little_list])
    test_run_1_twitter.append(result)


cleaned_results = [arr[0] for arr in test_run_1_twitter]

with open("twitter_test_predictions.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)

    # header row
    writer.writerow(["beleive", "disbelieve", "question", "mock"])

    # data rows
    for row in cleaned_results:
        writer.writerow(row)

