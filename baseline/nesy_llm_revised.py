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

beleif_ranking_list = []

for tweet in list_of_text:
    initial_prompt = 'You receive the following tweet: '
    #tweet
    second_part_prompt = '. Provide a decimal from 0 to 1 (where 0 is not at all and 1 is completely) for this tweet measuring how much its content beleives the following statement: '
    statement_coded = 'The earth is flat.'
    full_llm_prompt = initial_prompt + tweet + second_part_prompt + statement_coded

    llm_output_list = generate_text(current_model, full_llm_prompt)
    beleif_ranking_list.append(llm_output_list)

disbeleif_ranking_list = []

for tweet in list_of_text:
    initial_prompt = 'You receive the following tweet: '
    #tweet
    second_part_prompt = '. Provide a decimal from 0 to 1 (where 0 is not at all and 1 is completely) for this tweet measuring how much its content disbeleives the following statement: '
    statement_coded = 'The earth is flat.'
    full_llm_prompt = initial_prompt + tweet + second_part_prompt + statement_coded

    llm_output_list = generate_text(current_model, full_llm_prompt)
    disbeleif_ranking_list.append(llm_output_list)


question_ranking_list = []

for tweet in list_of_text:
    initial_prompt = 'You receive the following tweet: '
    #tweet
    second_part_prompt = '. Provide a decimal from 0 to 1 (where 0 is not at all and 1 is completely) for this tweet measuring how much its content questions the following statement: '
    statement_coded = 'The earth is flat.'
    full_llm_prompt = initial_prompt + tweet + second_part_prompt + statement_coded

    llm_output_list = generate_text(current_model, full_llm_prompt)
    question_ranking_list.append(llm_output_list)


mock_ranking_list = []

for tweet in list_of_text:
    initial_prompt = 'You receive the following tweet: '
    #tweet
    second_part_prompt = '. Provide a decimal from 0 to 1 (where 0 is not at all and 1 is completely) for this tweet measuring how much its content mocks the following statement: '
    statement_coded = 'The earth is flat.'
    full_llm_prompt = initial_prompt + tweet + second_part_prompt + statement_coded

    llm_output_list = generate_text(current_model, full_llm_prompt)
    mock_ranking_list.append(llm_output_list)


with open("revised_output.txt", "w") as file:
    file.write(str(beleif_ranking_list))
    file.write(str(disbeleif_ranking_list))
    file.write(str(question_ranking_list))
    file.write(str(mock_ranking_list))



def get_value(list):

    isolated_ranking_list = []

    for response in list:
        init_prompt = "You receive the following response ranking a tweet: "
        # response
        secondary_prompt = ". This response contains a decimal between 0 to 100 percent. Please isolate this decimal and output it between square brackets []. I.e. example outputs would be [0] or [.6] of [.05] or [1.0]."
        second_llm_prompt = init_prompt + response + secondary_prompt

        llm_second_output = generate_text(current_model, second_llm_prompt)
        isolated_ranking_list.append((llm_second_output))

    numerical_rankings = []

    for val in isolated_ranking_list:
        match = re.findall(r"\[(.*?)\]", val)
        numerical_rankings.append(match)

    return numerical_rankings



beleif_num_results = get_value(beleif_ranking_list)
disbeleif_num_results = get_value(disbeleif_ranking_list)
question_num_results = get_value(question_ranking_list)
mock_num_results = get_value(mock_ranking_list)



with open("revised_ranking_output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    
    # optional header
    writer.writerow(["beleif", "disbeleif", "question", "mock"])
    
    for i in range(len(beleif_num_results)):
        writer.writerow([
            beleif_num_results[i],
            disbeleif_num_results[i],
            question_num_results[i],
            mock_num_results[i]
        ])

##############################################################################################################################
