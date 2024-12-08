
import json
import pandas as pd
import spacy
from transformers import pipeline, AutoTokenizer
from collections import Counter
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

file_path_fr = '/Users/darianlee/PycharmProjects/entity_preprocessing/lookup_table_fr.json'
file_path_it = '/Users/darianlee/PycharmProjects/entity_preprocessing/lookup_table_it.json'
file_path_es = '/Users/darianlee/PycharmProjects/entity_preprocessing/lookup_table_es.json'
with open(file_path_fr, 'r', encoding='utf-8') as f:
    tables_data_fr = json.load(f)

print(tables_data_fr)

with open(file_path_it, 'r', encoding='utf-8') as f:
    tables_data_it = json.load(f)

print(tables_data_it)

with open(file_path_es, 'r', encoding='utf-8') as f:
    tables_data_es = json.load(f)

print(tables_data_es)


data_es = os.path.join(os.path.dirname(__file__), "/Users/darianlee/PycharmProjects/entity_preprocessing/es.jsonl")
df_es = pd.read_json(data_es, lines=True)


data_it = os.path.join(os.path.dirname(__file__), "/Users/darianlee/PycharmProjects/entity_preprocessing/it.jsonl")
df_it = pd.read_json(data_it, lines=True)


data_fr = os.path.join(os.path.dirname(__file__), "/Users/darianlee/PycharmProjects/entity_preprocessing/french.jsonl")
df_fr = pd.read_json(data_fr, lines=True)


device = 0 if torch.cuda.is_available() else -1
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)
df_es['source_no_punct'] = df_es['source'].apply(remove_punctuation)
df_es['target_no_punc'] = df_es['target'].apply(remove_punctuation)

df_it['source_no_punct'] = df_it['source'].apply(remove_punctuation)
df_it['target_no_punc'] = df_it['target'].apply(remove_punctuation)

df_fr['source_no_punct'] = df_fr['source'].apply(remove_punctuation)
df_fr['target_no_punc'] = df_fr['target'].apply(remove_punctuation)


def codeswitch_es(sentence):
        print("starting codeswitch")
        dictionary = tables_data_es['es']

        sorted_keys = sorted(dictionary.keys(), key=lambda k: len(k), reverse=True)

        # we need to do all this so that it matches with the biggest possible substring rather than a smaller one
        for key in sorted_keys:

            if not key.lower() in ["which", "alaska", "california", "what", "who", "why", "did", "w", "p", "whic", "of",
                                   "on the", "t", "o", "g", "the", "tri", "of the", "i"]:
                sentence = re.sub(r'\b' + re.escape(key) + r'\b', dictionary[key], sentence)

        return sentence


def codeswitch_fr(sentence):
    print("starting codeswitch")
    dictionary = tables_data_fr['fr']

    sorted_keys = sorted(dictionary.keys(), key=lambda k: len(k), reverse=True)

    # we need to do all this so that it matches with the biggest possible substring rather than a smaller one
    for key in sorted_keys:

        if not key.lower() in ["which", "alaska", "california", "what", "who", "why", "did", "w", "p", "whic", "of",
                               "on the", "t", "o", "g", "the", "tri", "of the", "i"]:

            sentence = re.sub(r'\b' + re.escape(key) + r'\b', dictionary[key], sentence)

    return sentence

def codeswitch_it(sentence):
        print("starting codeswitch")
        dictionary = tables_data_it['it']

        sorted_keys = sorted(dictionary.keys(), key=lambda k: len(k), reverse=True)

        # we need to do all this so that it matches with the biggest possible substring rather than a smaller one
        for key in sorted_keys:

            if not key.lower() in ["which", "alaska", "california", "what", "who", "why", "did", "w", "p", "whic", "of",
                                   "on the", "t", "o", "g", "the", "tri", "of the", "i"]:
                sentence = re.sub(r'\b' + re.escape(key) + r'\b', dictionary[key], sentence)

        return sentence




df_es['code_switch'] = df_es['source_no_punct'].apply(codeswitch_es)
df_it['code_switch'] = df_it['source_no_punct'].apply(codeswitch_it)
df_fr['code_switch'] = df_fr['source_no_punct'].apply(codeswitch_fr)

df_es.to_csv('/Users/darianlee/PycharmProjects/entity_preprocessing/df_es.csv', index=False, encoding='utf-8')
df_it.to_csv('/Users/darianlee/PycharmProjects/entity_preprocessing/df_it.csv', index=False, encoding='utf-8')
df_fr.to_csv('/Users/darianlee/PycharmProjects/entity_preprocessing/df_fr.csv', index=False, encoding='utf-8')



