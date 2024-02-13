import string

def keep_valid_characters(line):
    valid_characters = string.ascii_letters + string.digits + string.punctuation + '\t\n '
    return ''.join(char for char in line if char in valid_characters)

def clean_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            cleaned_line = keep_valid_characters(line)
            outfile.write(cleaned_line)

# Example usage:
input_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/en-eu/train_en-eu_long.txt'
output_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/en-eu/train_en-eu_long_cleaned.txt'

clean_file(input_file_path, output_file_path)

input_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/eu-en/train_eu-en_long.txt'
output_file_path = '/ikerlariak/jiblaing/NMT_2023/M2M_finetuning_v2/corpora/eu-en/train_eu-en_long_cleaned.txt'

clean_file(input_file_path, output_file_path)