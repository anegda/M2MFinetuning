import json

# Function to read lines from a text file and create JSON
import json

# Function to read lines from a text file and create JSON
def create_json(input_file, output_file, src, tgt):
    data_list = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            l1, l2 = line.strip().split('\t')
            data_list.append({'translation': {src: l1, tgt: l2}})

    # Create a dictionary with 'data' key containing the list of dictionaries
    data_dict = {'data': data_list}

    # Write the dictionary as JSON to a file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, indent=2)

# File paths
input_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/en-eu/train_en-eu_long_cleaned.txt'
output_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/en-eu/train_en-eu_long.json'
create_json(input_file_path, output_file_path, "en", "eu")

input_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/en-eu/development_en-eu.txt'
output_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/en-eu/development_en-eu.json'
#create_json(input_file_path, output_file_path, "en", "eu")

input_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/eu-en/train_eu-en_cleaned.txt'
output_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/eu-en/train_eu-en_long.json'
create_json(input_file_path, output_file_path, "eu", "en")

input_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/eu-en/development_eu-en.txt'
output_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/eu-en/development_eu-en.json'
#create_json(input_file_path, output_file_path, "eu", "en")
