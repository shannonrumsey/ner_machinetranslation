
import argparse
def main():
    import re
    from unidecode import unidecode
    import pandas as pd
    import random
    import json
    from datasets import load_dataset
    import requests
    import sentencepiece as spm
    parser = argparse.ArgumentParser(description="Get the file path")


    parser.add_argument("english_squad_path", type=str, help="Path to the english_squad data")
    parser.add_argument("french_csv", type=str, help="Path to the French CSV file")
    parser.add_argument("italian_csv", type=str, help="Path to the Italian CSV file")
    parser.add_argument("spanish_csv", type=str, help="Path to the Spanish CSV file")
    parser.add_argument("csv_dir", type=str, help="Directory to save CSV files")


    args = parser.parse_args()


    csv_dir = args.csv_dir
    args = parser.parse_args()
    english_squad_path = args.english_squad_path
    french_file_path = args.french_csv
    italian_file_path = args.italian_csv
    spanish_file_path = args.spanish_csv


    def download_file(url, filename):
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

    download_file("https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/master/SQuAD-es-v2.0/train-v2.0-es.json", "train-v2.0-es.json")
    download_file("https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/master/SQuAD-es-v2.0/dev-v2.0-es.json", "dev-v2.0-es.json")

    with open("train-v2.0-es.json", "r", encoding="utf-8") as train_file:
        train_data = json.load(train_file)

    with open("dev-v2.0-es.json", "r", encoding="utf-8") as dev_file:
        dev_data = json.load(dev_file)

    print("Train Data Example:")
    print(json.dumps(train_data["data"][0], indent=4, ensure_ascii=False))

    with open("train-v2.0-es.json", "r", encoding="utf-8") as train_file:
        train_data = json.load(train_file)

    questions_es = []

    for entry in train_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                questions_es.append(qa["question"])

    for question in questions_es[:10]:
        print(question)

    def clean_text(text):
        text = re.sub(r"\bi\b", "I", text)
        text = re.sub(r"[^\w\s¿?.]", "", text)
        text = unidecode(text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text.strip())
        return text



    with open(english_squad_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data["data"]
    questions_en = []
    for item in data:
        for paragraph in item['paragraphs']:
            for qas in paragraph['qas']:
                questions_en.append(qas['question'])

    print(questions_en[:5])

    dataset = load_dataset("qwant/squad_fr", trust_remote_code=True)
    train_data = dataset["train"]

    dataset_it = load_dataset("squad_it")
    # cause italian dataset is tiny
    from datasets import concatenate_datasets

    train_data_it = concatenate_datasets([dataset_it["train"], dataset_it["test"]])

    questions_fr = [example['question'] for example in train_data]
    questions_it = [example['question'] for example in train_data_it]

    print("First 5 questions from SQuAD-fr:", questions_fr[:5])
    print("First 5 questions from SQuAD-it:", questions_it[:5])

    print("Number of questions in English dataset:", len(questions_en))
    print("Number of questions in French dataset:", len(questions_fr))
    print("Number of questions in Spanish dataset:", len(questions_es))
    print("Number of questions in Italian dataset:", len(questions_it))

    def generate_corpus(lst):


        corpus = " ".join(lst)
        return corpus



    corpus_en = generate_corpus(questions_en)
    corpus_fr = generate_corpus(questions_fr)
    corpus_es = generate_corpus(questions_es)
    corpus_it = generate_corpus(questions_it)

    corpus_en = clean_text(corpus_en)
    corpus_fr = clean_text(corpus_fr)
    corpus_es = clean_text(corpus_es)
    corpus_it = clean_text(corpus_it)

    french = pd.read_csv(french_file_path)
    italian = pd.read_csv(italian_file_path)
    spanish = pd.read_csv(spanish_file_path)

    important_columns = ["code_switch", "target"]

    french = french[important_columns]
    italian = italian[important_columns]
    spanish = spanish[important_columns]

    french[important_columns] = french[important_columns].applymap(clean_text)
    italian[important_columns] = italian[important_columns].applymap(clean_text)
    spanish[important_columns] = spanish[important_columns].applymap(clean_text)


    all_corpora_text = "\n".join([corpus_en, corpus_fr, corpus_es, corpus_it])


    french_text = "\n".join(french["code_switch"].astype(str) + " " + french["target"].astype(str))
    italian_text = "\n".join(italian["code_switch"].astype(str) + " " + italian["target"].astype(str))
    spanish_text = "\n".join(spanish["code_switch"].astype(str) + " " + spanish["target"].astype(str))


    combined_text = "\n".join([all_corpora_text, french_text, italian_text, spanish_text])


    with open("combined_corpus.txt", "w", encoding="utf-8") as f:
        f.write(combined_text)
    with open('combined_corpus.txt', 'r', encoding='utf-8') as file:
        text = file.read()


    tokens = text.split()


    unique_tokens = set(tokens)

    unique_token_count = len(unique_tokens)

    print("Total unique token count👚👚👚👚👚 :", unique_token_count)

    spm.SentencePieceTrainer.train(input="combined_corpus.txt", model_prefix="tokenizer_combined", vocab_size=30000, character_coverage=0.9995, model_type="bpe")




    def apply_bpe_tokenizer(df, column_name, model_prefix="tokenizer_combined"):
        sp = spm.SentencePieceProcessor(model_file="tokenizer_combined.model")
        df[column_name] = df[column_name].apply(lambda text: sp.encode(text, out_type=str))
        return df

    print("======= BEFORE =======")
    print(corpus_en[:1000])
    print(corpus_es[:1000])
    print(corpus_it[:1000])
    print(corpus_fr[:1000])

    sp = spm.SentencePieceProcessor(model_file="tokenizer_combined.model")
    corpus_es = sp.encode(corpus_es, out_type=str)
    corpus_en = sp.encode(corpus_en, out_type=str)
    corpus_it = sp.encode(corpus_it, out_type=str)
    corpus_fr = sp.encode(corpus_fr, out_type=str)

    print("======= AFTER =======")
    print(corpus_en[:100])
    print(corpus_es[:100])
    print(corpus_it[:100])
    print(corpus_fr[:100])


    tokenizers = {}
    for column in important_columns:
        print(f"Processing column: {column}")
        tokenizers[column] = {
            "french": apply_bpe_tokenizer(french, column),
            "italian": apply_bpe_tokenizer(italian, column),
            "spanish": apply_bpe_tokenizer(spanish, column)
        }


    import os





    os.makedirs(csv_dir, exist_ok=True)


    for column, languages in tokenizers.items():
        for language, tokens in languages.items():
            # Check if the tokens is a DataFrame and save it
            if isinstance(tokens, pd.DataFrame):
                csv_file_path = os.path.join(csv_dir, f'{column}_{language}.csv')
                tokens.to_csv(csv_file_path, index=False)
                print(f'Saved tokenized data for {column} in {language} to {csv_file_path}')

    def create_blocks(corpus, code):
        blocks = []


        while len(corpus) > 30:
            block = [code]
            new = corpus[:30]
            block = block + new
            blocks.append(block)
            corpus = corpus[30:]
        return blocks

    blocks_it = create_blocks(corpus_it, "<it>")
    blocks_es = create_blocks(corpus_es, "<es>")
    blocks_en = create_blocks(corpus_en, "<en>")
    blocks_fr = create_blocks(corpus_fr, "<fr>")

    print(blocks_it[:5])
    print(blocks_es[:5])
    print(blocks_en[:5])
    print(blocks_fr[:5])


    min_len = min([len(blocks_it), len(blocks_es), len(blocks_en), len(blocks_fr)])
    blocks_it = blocks_it[:min_len]
    blocks_en = blocks_en[:min_len]
    blocks_es = blocks_es[:min_len]
    blocks_fr = blocks_fr[:min_len]

    df = pd.DataFrame({
        'it': blocks_it,
        'en': blocks_en,
        'es': blocks_es,
        'fr': blocks_fr
    })


    df.to_csv(csv_dir/'blocks.csv', index=False)

if __name__ == "__main__":
    main()


