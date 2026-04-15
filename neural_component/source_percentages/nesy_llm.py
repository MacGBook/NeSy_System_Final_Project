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

with open("flat_earth_tweets.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        list_of_text.append(row["text"])

#print(len(list_of_text))

##############################################################################################################################
# Use one LLM pull to get the list of sources

initial_prompt = 'You receive the following 752 tweets, stored as strings in a list: '
kaggle_data = str(list_of_text)
second_part_prompt = '. Based on this data, make an exhaustive list of all sources (such as news outlets, scholars, institutions, buisnesses, universities, podcasts, major websites, and influencers) mentioned in these tweets.'
full_llm_prompt = initial_prompt + kaggle_data + second_part_prompt

#llm_output_list = generate_text(current_model, full_llm_prompt)

#with open("output.txt", "a") as file:
    #file.write(llm_output_list)

##############################################################################################################################
# Use another LLM pull to get percentages from curated list of sources

list_of_sources_from_first_run = ["Associated Press (AP)", "Reuters”, “BBC News”, “CNN", "The Guardian", "Bloomberg", "New York Times", "Fox News", "The Washington Post", "NASA (National Aeronautics and Space Administration)", "NSF (National Science Foundation)", "The Black Goo- (claimed to be a researcher)", "The United States Geological Survey (USGS)", "Stanford University", "The Philosophical Society of America", "Philosophical Studies at Harvard", "The Stanford University Philosophy Center", "The University of Cambridge", "ItsToddLove", "A_P_S”, “Wikipedia", "Reddit"]

second_ip = 'You receive the following dataset of 752 tweets: '
#kaggle_data
next_prompt = '. Based on the content of these tweets, assign a probability from 0 to 100 percent to each of these sources: '
list_of_them_sources = str(list_of_sources_from_first_run)
statement = '...as to how much they agree with the statment: the earth is flat.'
reinforcement = 'there needs to be a probability listed for each of the sources mentioned in this list!'
second_full_llm_prompt = second_ip + kaggle_data + next_prompt + list_of_them_sources + ". " + statement + reinforcement

#llm_second_output = generate_text(current_model, second_full_llm_prompt)

#with open("second_output.txt", "w") as file:
    #file.write(llm_second_output)
