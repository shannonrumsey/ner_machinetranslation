
import json
import pandas as pd
import spacy
from transformers import pipeline, AutoTokenizer
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re
import argparse

def main():
    # getting command-line arguments for user input
    parser = argparse.ArgumentParser(description="Process code-switching for multiple languages.")
    parser.add_argument('--lookup_fr', type=str, required=True, help="Path to the French lookup table JSON file")
    parser.add_argument('--lookup_it', type=str, required=True, help="Path to the Italian lookup table JSON file")
    parser.add_argument('--lookup_es', type=str, required=True, help="Path to the Spanish lookup table JSON file")
    parser.add_argument('--data_fr', type=str, required=True, help="Path to the French data JSONL file")
    parser.add_argument('--data_it', type=str, required=True, help="Path to the Italian data JSONL file")
    parser.add_argument('--data_es', type=str, required=True, help="Path to the Spanish data JSONL file")
    parser.add_argument('--output_fr', type=str, required=True, help="Path to save the processed French output CSV")
    parser.add_argument('--output_it', type=str, required=True, help="Path to save the processed Italian output CSV")
    parser.add_argument('--output_es', type=str, required=True, help="Path to save the processed Spanish output CSV")

    args = parser.parse_args()

    # loading lookup tables from command line
    with open(args.lookup_fr, 'r', encoding='utf-8') as f:
        tables_data_fr = json.load(f)

    with open(args.lookup_it, 'r', encoding='utf-8') as f:
        tables_data_it = json.load(f)

    with open(args.lookup_es, 'r', encoding='utf-8') as f:
        tables_data_es = json.load(f)

    # loading data from comand line
    df_es = pd.read_json(args.data_es, lines=True)
    df_it = pd.read_json(args.data_it, lines=True)
    df_fr = pd.read_json(args.data_fr, lines=True)

    device = 0 if torch.cuda.is_available() else -1

    def remove_punctuation(text):
        text = re.sub(r'\?', ' ?', text)  # add a space before the question mark
        return re.sub(r'[^\w\s\?]', '', text)  # remove all punctuation except question mark

    df_es['source_no_punct'] = df_es['source'].apply(remove_punctuation)
    df_es['target_no_punc'] = df_es['target'].apply(remove_punctuation)

    df_it['source_no_punct'] = df_it['source'].apply(remove_punctuation)
    df_it['target_no_punc'] = df_it['target'].apply(remove_punctuation)

    df_fr['source_no_punct'] = df_fr['source'].apply(remove_punctuation)
    df_fr['target_no_punc'] = df_fr['target'].apply(remove_punctuation)

    def add_underscores(lookup_table):
        return {key: "_".join(value.split()) for key, value in lookup_table.items()}

    def codeswitch_es(sentence):
        print("Starting code-switching for Spanish")
        dictionary = tables_data_es['es']
        dictionary = add_underscores(dictionary)

        sorted_keys = sorted(dictionary.keys(), key=lambda k: len(k), reverse=True)
        for key in sorted_keys:
            if not key.lower() in ["which", "alaska", "california", "what", "who", "why", "did", "w", "p", "whic", "of", "on the", "t", "o", "g", "the", "tri", "of the", "i"]:
                sentence = re.sub(r'\b' + re.escape(key) + r'\b', dictionary[key], sentence)

        return sentence

    def codeswitch_fr(sentence):
        print("Starting code-switching for French")
        dictionary = tables_data_fr['fr']
        dictionary = add_underscores(dictionary)

        sorted_keys = sorted(dictionary.keys(), key=lambda k: len(k), reverse=True)
        for key in sorted_keys:
            if not key.lower() in ["which", "alaska", "california", "what", "who", "why", "did", "w", "p", "whic", "of", "on the", "t", "o", "g", "the", "tri", "of the", "i"]:
                sentence = re.sub(r'\b' + re.escape(key) + r'\b', dictionary[key], sentence)

        return sentence

    def codeswitch_it(sentence):
        print("Starting code-switching for Italian")
        dictionary = tables_data_it['it']
        dictionary = add_underscores(dictionary)

        sorted_keys = sorted(dictionary.keys(), key=lambda k: len(k), reverse=True)
        for key in sorted_keys:
            if not key.lower() in ["which", "alaska", "california", "what", "who", "why", "did", "w", "p", "whic", "of", "on the", "t", "o", "g", "the", "tri", "of the", "i"]:
                sentence = re.sub(r'\b' + re.escape(key) + r'\b', dictionary[key], sentence)

        return sentence

    # apply code-switching
    df_es['code_switch'] = df_es['source_no_punct'].apply(codeswitch_es)
    df_it['code_switch'] = df_it['source_no_punct'].apply(codeswitch_it)
    df_fr['code_switch'] = df_fr['source_no_punct'].apply(codeswitch_fr)

    # save to CSV
    df_es.to_csv(args.output_es, index=False, encoding='utf-8')
    df_it.to_csv(args.output_it, index=False, encoding='utf-8')
    df_fr.to_csv(args.output_fr, index=False, encoding='utf-8')

    print(f"Files successfully saved:\n{args.output_es}\n{args.output_it}\n{args.output_fr}")

if __name__ == "__main__":
    main()




