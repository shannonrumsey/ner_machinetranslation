

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
tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

data_es = os.path.join(os.path.dirname(__file__), "/Users/darianlee/PycharmProjects/entity_preprocessing/es.jsonl")
df_es = pd.read_json(data_es, lines=True)


data_it = os.path.join(os.path.dirname(__file__), "/Users/darianlee/PycharmProjects/entity_preprocessing/it.jsonl")
df_it = pd.read_json(data_it, lines=True)


data_fr = os.path.join(os.path.dirname(__file__), "/Users/darianlee/PycharmProjects/entity_preprocessing/french.jsonl")
df_fr = pd.read_json(data_fr, lines=True)


device = 0 if torch.cuda.is_available() else -1

import wikipediaapi
import wikipedia


def translate_entity(entity_name, language_code):
    # user_agent is just something wiki requires to know who's making the request
    user_agent = "Darian"
    if entity_name == "on the":
        return "on the"
    if entity_name == "Grey":
        return "Grey"

    if entity_name == "Best Actor":  # just some small mistakes that annoy me
        if language_code == "fr":
            return "Meilleur_Acteur"
        elif language_code == "es":
            return "mejor_Actor"
        elif language_code == "it":
            return "Miglior_Attore"
    if entity_name == "US":
        entity_name = "United States"
    if entity_name == "##ja Cat":
        entity_name = "Doja_Cat"


    # initialize wiki object for english
    wiki_en = wikipediaapi.Wikipedia('en', headers={'User-Agent': user_agent})

    try:
        # search for similar pages (I stopped using the exact page because often it wasn't found or was wrong)
        search_results = wikipedia.search(entity_name, results=1)

        if search_results:
            # get the first search result (most relevant)
            most_relevant_entity = search_results[0]
            page_en = wiki_en.page(most_relevant_entity)

            # if the page exists, try to translate
            if page_en.exists():
                if language_code in page_en.langlinks:
                    # get the foreign page and its title
                    foreign_page = page_en.langlinks[language_code]
                    title = foreign_page.title

                    # remove parentheses and their content using regex. (this is to avoid getting outputs like Titanic (película)
                    title = re.sub(r'\s?\(.*?\)', '', title)

                    # replace spaces with underscores in the title
                    return title.replace(" ", "_")
                else:
                    return entity_name.replace(" ", "_")
            else:
                return entity_name.replace(" ", "_")
        else:
            return entity_name.replace(" ", "_")
    except Exception as e:

        return entity_name.replace(" ", "_")

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

df_es['source_no_punct'] = df_es['source'].apply(remove_punctuation)
df_es['target_no_punc'] = df_es['target'].apply(remove_punctuation)
es_es = df_es['target_no_punc']
es_en = df_es['source_no_punct']

df_it['source_no_punct'] = df_it['source'].apply(remove_punctuation)
df_it['target_no_punc'] = df_it['target'].apply(remove_punctuation)
it_it = df_it['target_no_punc']
it_en = df_it['source_no_punct']


df_fr['source_no_punct'] = df_fr['source'].apply(remove_punctuation)
df_fr['target_no_punc'] = df_fr['target'].apply(remove_punctuation)
fr_fr = df_fr['target_no_punc']
fr_en = df_fr['source_no_punct']

class Lookup_table:
    def __init__(self):

            # Spanish (es)
            self.es = {}
            self.es_issues_rows = []
            self.es_undecided = set()
            self.es_issues_entities = {}

            # Italian (it)
            self.it = {}
            self.it_issues_rows = []
            self.it_undecided = set()
            self.it_issues_entities = {}

            # French (fr)
            self.fr = {}
            self.fr_issues_rows = []
            self.fr_undecided = set()
            self.fr_issues_entities = {}
    def get_table(self, lang_code, lang, en, model):
        print("\n\n\n ================ GETTING TABLE FOR ", lang_code, "====================== \n\n\n")
        if lang_code == "es":
            language_data = self.es
            language_issues_rows = self.es_issues_rows
            language_undecided = self.es_undecided
            language_issues_entities = self.es_issues_entities
        elif lang_code == "it":
            language_data = self.it
            language_issues_rows = self.it_issues_rows
            language_undecided = self.it_undecided
            language_issues_entities = self.it_issues_entities
        elif lang_code == "fr":
            language_data = self.fr
            language_issues_rows = self.fr_issues_rows
            language_undecided = self.fr_undecided
            language_issues_entities = self.fr_issues_entities
        else:
            raise ValueError(f"Unsupported language code: {lang_code}")

        # Pass these variables into another function
        self.populate_table(language_data, language_issues_rows, language_undecided, language_issues_entities, lang, en, model, lang_code)


    def populate_table(self, language_data, language_issues_rows, language_undecided, language_issues_entities, lang, en, model, lang_code):

        nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=device, aggregation_strategy="simple")
        for index in range(len(lang)):
            # merge subwords function is used because once in awhile it will try to do subwords and get confused. this is to prevent that
            # this only happened 50 times in the data, but still its good to take out
            ner_results_en = self.merge_subwords_in_ner_results(nlp(en[index]))
            ner_results_lang = self.merge_subwords_in_ner_results(nlp(lang[index]))
            if len(ner_results_en)  != len(ner_results_lang):

                language_issues_rows.append(index) ## change this later ## this row has a potential issue
                for result in ner_results_en:
                    if result['entity_group'] != 'PER': # people will likely have the same name cross linguistically

                        if result["word"] not in language_data:

                            language_undecided.add(result["word"]) # to keep track of possibly untranslated entities

                continue # do not swap in this case

            else:
                for result_index in range(len(ner_results_en)):
                    if ner_results_en[result_index]["entity_group"] != ner_results_lang[result_index]["entity_group"]:
                        #there is likely a misallignment
                        if ner_results_en[result_index]['entity_group'] != 'PER': # again, wiki has a tendency to mistranslate the persons
                            language_undecided.add(ner_results_en[result_index]["word"])
                            continue
                    else:
                        en_ent = ner_results_en[result_index]["word"]
                        lang_ent = "_".join(ner_results_lang[result_index]["word"].split())




                    if en_ent in language_data: # possible mistranslation of entity

                        if language_data[en_ent] != lang_ent:
                            if en_ent in language_issues_entities:
                                language_issues_entities[en_ent].append(lang_ent) #add the other possible translation
                            else:
                                language_issues_entities[en_ent] = [language_data[en_ent], lang_ent] # add both
                    else:
                        language_data[en_ent] = lang_ent

        for ent_key in language_issues_entities: # replace rows with multiple translations as the most commonly occuring one

            ent = self.most_frequent_translation(language_issues_entities[ent_key])
            if ent is not None:
                print("max ent found! :", ent_key, " -> ", ent)
                language_data[ent_key] = ent
            if ent is None:
                ent = translate_entity(ent_key, lang_code) # if the model is undecided, translate it using wiki data
                language_data[ent_key] = ent
                print("got ent from wikidata :", ent_key, " -> ", ent)
        self.translate_undecided(language_undecided, language_data, lang_code)
    # every once in awhile it will try to do subwords and get confused. this is to prevent that
    # this only happened 50 times in the data, but still its good to take out
    def merge_subwords_in_ner_results(self, ner_results):
        merged_results = []
        current_word = ""
        current_entity_group = None
        current_score = None

        for result in ner_results:
            word = result["word"]
            entity_group = result["entity_group"]
            score = result["score"]

            # if the word starts with "##", it is a continuation of the previous word
            if word.startswith("##"):
                current_word += word[2:]  # remove the "##" and append the rest
            else:
                # If there's a previous word being built, append it to the merged results
                if current_word:
                    merged_results.append({
                        "word": current_word,
                        "entity_group": current_entity_group,
                        "score": current_score
                    })
                # start a new word
                current_word = word
                current_entity_group = entity_group
                current_score = score

        # append the last word after the loop
        if current_word:
            merged_results.append({
                "word": current_word,
                "entity_group": current_entity_group,
                "score": current_score
            })

        return merged_results
    def most_frequent_translation(self, lst):

        counter = Counter(lst)

        most_common_element, most_common_count = counter.most_common(1)[0]


        if most_common_count >= 2:
            return most_common_element
        else:
            return None

    def translate_undecided(self, language_undecided, language_data, lang_code):
        print("going through undecided elements now")
        missing_elements = language_undecided - set(language_data.keys())
        for el in missing_elements:
            ent = translate_entity(el, lang_code)  # if the model is undecided, translate it using wiki data
            language_data[el] = ent
            print("got undecided ent from wikidata :", el, " -> ", ent)




table = Lookup_table()
table.get_table("es", es_es, es_en, model)
print(table.es)
tables_data_es = {
    "es": table.es,
}
import json
with open('/Users/darianlee/PycharmProjects/entity_preprocessing/lookup_table_es.json', 'w', encoding='utf-8') as f:
    json.dump(tables_data_es, f, ensure_ascii=False, indent=4)

table.get_table("it", it_it, it_en, model)
print(table.it)
tables_data_it = {
    "it": table.it,
}
with open('/Users/darianlee/PycharmProjects/entity_preprocessing/lookup_table_it.json', 'w', encoding='utf-8') as f:
    json.dump(tables_data_it, f, ensure_ascii=False, indent=4)


table.get_table("fr", fr_fr, fr_en, model)
print(table.fr)
tables_data_fr = {
    "fr": table.fr,
}
with open('/Users/darianlee/PycharmProjects/entity_preprocessing/lookup_table_fr.json', 'w', encoding='utf-8') as f:
    json.dump(tables_data_fr, f, ensure_ascii=False, indent=4)










