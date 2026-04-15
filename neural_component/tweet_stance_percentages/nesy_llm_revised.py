# IMPORTS

import csv

#General LLM
import json
import ollama
import re 

##############################################################################################################################

current_model = 'gemma3:1b'

#LLM SET UP
# Ollama: function for basic text generation
def generate_text(model_name, prompt):
    response = ollama.generate(model=model_name, prompt=prompt)
    return(f"Generated text ({model_name}):\n{response['response']}\n")

##############################################################################################################################

list_of_text = []

with open("tweet_test.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        list_of_text.append(str(row))

##############################################################################################################################
# Use one LLM pull to get the list of sources

ranking_list = []

for tweet in list_of_text:
    initial_prompt = 'You receive the following tweet: '
    #tweet
    second_part_prompt = '. Provide a percentage from 0 to 100 percent for this tweet measuring how much its content agrees with the following statement: '
    statement_coded = 'The earth is flat.'
    full_llm_prompt = initial_prompt + tweet + second_part_prompt + statement_coded

    llm_output_list = generate_text(current_model, full_llm_prompt)
    ranking_list.append(llm_output_list)

with open("revised_output.txt", "w") as file:
    file.write(str(ranking_list))


isolated_ranking_list = []

for response in ranking_list:
    init_prompt = "You receive the following response ranking a tweet: "
    # response
    secondary_prompt = ". This response contains a percentage between 0 to 100 percent. Please isolate this percentage and output it between square brackets []. I.e. example outputs would be [0%] or [60%] of [5%] or [100%]."
    second_llm_prompt = init_prompt + response + secondary_prompt

    llm_second_output = generate_text(current_model, second_llm_prompt)
    isolated_ranking_list.append(llm_second_output)

numerical_rankings = []

for val in isolated_ranking_list:
    match = re.findall(r"\[(.*?)\]", val)
    numerical_rankings.append(match)


with open("revised_ranking_output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    
    for item in range(len(list_of_text)):
        writer.writerow([list_of_text[item]])  # write each item as a row (1 column)
        writer.writerow([numerical_rankings[item]])




##############################################################################################################################
