import pandas as pd
import sentencepiece as spm
import os
import pandas as pd
import ast

folder_path = '/Users/darianlee/PycharmProjects/entity_preprocessing/csv_files'
import torch


csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]


dfs = []
device = torch.device("cpu")

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df_name = os.path.splitext(csv_file)[0]
    globals()[df_name] = pd.read_csv(file_path)
    dfs.append(globals()[df_name])


    lang = df_name.split('_')[-1]


    globals()[df_name].rename(columns={
        "code_switch": f"code_switch_{lang}",
        "target": f"target_{lang}"
    }, inplace=True)


    for column in [f"code_switch_{lang}", f"target_{lang}"]:
        def ensure_list_of_strings(value):
            if isinstance(value, str):
                try:

                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                        return parsed
                except (ValueError, SyntaxError):
                    pass

                return [value]
            elif isinstance(value, list):

                return [str(item) for item in value]
            else:

                return [str(value)]

        globals()[df_name][column] = globals()[df_name][column].apply(ensure_list_of_strings)


    print(f"Info for {df_name}:")
    print(f"Missing values:\n{globals()[df_name].isna().sum()}\n")
    print(f"Data types:\n{globals()[df_name].dtypes}\n")

    print(f"Head of {df_name}:")
    print(globals()[df_name].head())
    print("\n")



pretrain = pd.read_csv("blocks.csv")

pretrain = pretrain[:20000]

import torch

import torch


"""if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)} (CUDA)")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("MPS is available. Using Apple GPU (M1/M2 chip).")
    device = torch.device("mps")
else:
    print("🔴🔴🔴 GPU is not available. Using CPU instead. 🔴🔴🔴")
    device = torch.device("cpu")

print(f"Using device: {device}")"""

print("pretrain shape: ", pretrain.shape)

device = torch.device("cpu") # I needed to add this in for comet


def ensure_list_of_strings(value):
    if isinstance(value, str):
        try:

            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return parsed
        except (ValueError, SyntaxError):
            pass

        return [value]
    elif isinstance(value, list):

        return [str(item) for item in value]
    else:

        return [str(value)]


for column in pretrain.columns:
    pretrain[column] = pretrain[column].apply(ensure_list_of_strings)


print("Head of pretrain after conversion:")
print(pretrain.head())
print("\nColumn types after conversion:")
print(pretrain.dtypes)


dfs.append(pretrain)

print(pretrain["en"][:5])
print(pretrain["es"][:5])
big_list_o_data = []

for df in dfs:
    for column in df.columns:
        big_list_o_data.append(df[column].tolist())


import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
# define dataset

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
# define dataset

print("defining the dataset")

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
# define dataset

print("defining the dataset")
class TranslationDataset(Dataset):
    def __init__(self, f_data, i_data, s_data, data, vocab_data = [], token_vocab=None, make_vocab=True, pretrain = True):
        if make_vocab: # this will be done once using the train and pretrain data, which is passed in in the vocab_data argument
            self.token_vocab = {"<PAD>": 0, "<unk>": 1, "-->" : 2, "<END>" : 3}
            for column_list in vocab_data:
                for row_list in column_list:

                    for token in row_list:
                        if token not in self.token_vocab:
                            self.token_vocab[token] = len(self.token_vocab)




        else:

            assert token_vocab is not None
            self.token_vocab = token_vocab
        if not pretrain:
            self.inverse_vocab = {index: token for token, index in self.token_vocab.items()}

            self.corpus_x_ids = []
            self.corpus_y_ids = []
            self.corpus_y_mask = [] # so that we are only computing the loss for the actual translation


            for source, target in zip(f_data["code_switch_french"], f_data["target_french"]):
                print(source, target)
                # concatenate source and target with <SEP> token in between
                x_ids = [self.token_vocab["<en>"]] + [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in source] + [self.token_vocab['-->']] + [self.token_vocab['<fr>']] + [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in target]
                y_ids = x_ids[1:] + [self.token_vocab['<END>']]
                # 0 for the source. 0 for sep. 1 for all the target ones, zero for end
                # find the index of the <SEP> token (2) in x_id
                sep_idx = x_ids.index(self.token_vocab['<fr>'])

                # create y_mask
                # - 0 for tokens up to and including <fr>
                # - 1 for target tokens
                # - 0 for the <END> token
                y_mask = [0] * (sep_idx) + [1] * (len(y_ids)+1 - sep_idx - 2) + [0]
                self.corpus_y_ids.append(torch.tensor(y_ids))
                self.corpus_y_mask.append(torch.tensor(y_mask))
                self.corpus_x_ids.append(torch.tensor(x_ids))

            for source, target in zip(s_data["code_switch_spanish"], s_data["target_spanish"]):
                print(source, target)
                # concatenate source and target with <SEP> token in between
                x_ids = [self.token_vocab["<en>"]] + [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in source] + [self.token_vocab['-->']] + [self.token_vocab['<es>']] + [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in target]
                y_ids = x_ids[1:] + [self.token_vocab['<END>']]                # 0 for the source. 0 for sep. 1 for all the target ones, zero for end
                # find the index of the <SEP> token (2) in x_id
                sep_idx = x_ids.index(self.token_vocab['<es>'])

                # create y_mask
                # - 0 for tokens up to and including <fr>
                # - 1 for target tokens
                # - 0 for the <END> token
                y_mask = [0] * (sep_idx) + [1] * (len(y_ids)+1 - sep_idx - 2) + [0]
                self.corpus_y_ids.append(torch.tensor(y_ids))
                self.corpus_y_mask.append(torch.tensor(y_mask))
                self.corpus_x_ids.append(torch.tensor(x_ids))

            for source, target in zip(i_data["code_switch_italian"], i_data["target_italian"]):
                print(source, target)
                # concatenate source and target with <SEP> token in between
                x_ids = [self.token_vocab["<en>"]] + [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in source] + [self.token_vocab['-->']] + [self.token_vocab['<it>']] + [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in target]
                y_ids = x_ids[1:] + [self.token_vocab['<END>']]                # 0 for the source. 0 for sep. 1 for all the target ones, zero for end
                # find the index of the <SEP> token (2) in x_id
                sep_idx = x_ids.index(self.token_vocab['<it>'])

                # create y_mask
                # - 0 for tokens up to and including <fr>
                # - 1 for target tokens
                # - 0 for the <END> token
                y_mask = [0] * (sep_idx) + [1] * (len(y_ids)+1 - sep_idx - 2) + [0]
                self.corpus_y_ids.append(torch.tensor(y_ids))
                self.corpus_y_mask.append(torch.tensor(y_mask))
                self.corpus_x_ids.append(torch.tensor(x_ids))



            for i, (x, y, mask) in enumerate(zip(self.corpus_x_ids, self.corpus_y_ids, self.corpus_y_mask)):
                """print(f"Index {i}:")
                print(f"size: {x.size(0)}")
                print(f"size: {y.size(0)}")
                print(f"size: {mask.size(0)}")"""

                assert x.size(0) == y.size(0) == mask.size(0), (
                    f"Mismatch in tensor sizes! "
                    f"x size: {x.size(0)}, y size: {y.size(0)}, mask size: {mask.size(0)}"
                )
            c = 1

            for i, (x, y, mask) in enumerate(zip(self.corpus_x_ids, self.corpus_y_ids, self.corpus_y_mask)):
                while c < 5:
                    print(x)
                    print(y)
                    print(mask)
                    c+=1
            paired_data = list(zip(self.corpus_x_ids, self.corpus_y_ids, self.corpus_y_mask))
            random.shuffle(paired_data)
            self.corpus_x_ids, self.corpus_y_ids, self.corpus_y_mask = zip(*paired_data)
            self.corpus_x_ids, self.corpus_y_ids, self.corpus_y_mask = list(self.corpus_x_ids), list(self.corpus_y_ids), list(self.corpus_y_mask)
        if pretrain:
            self.inverse_vocab = {index: token for token, index in self.token_vocab.items()}

            self.corpus_x_ids = []
            self.corpus_y_ids = []
            self.corpus_y_mask = None # this wont be used for pretrain

            for column in ["it", "en", "es", "fr"]:
                for row in data[column]:
                    x_ids = [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in row]
                    y_ids = x_ids[1:] + [self.token_vocab['<END>']]

                    self.corpus_x_ids.append(torch.tensor(x_ids))
                    self.corpus_y_ids.append(torch.tensor(y_ids))
            paired_data = list(zip(self.corpus_x_ids, self.corpus_y_ids))
            random.shuffle(paired_data)
            self.corpus_x_ids, self.corpus_y_ids = zip(*paired_data)
            self.corpus_x_ids, self.corpus_y_ids = list(self.corpus_x_ids), list(self.corpus_y_ids)

            self.corpus_y_mask = self.corpus_x_ids # this will be ignored. we just need it to be indexable so we dont get errors










    def __len__(self):
        return len(self.corpus_x_ids)

    def __getitem__(self, idx):
        return self.corpus_x_ids[idx], self.corpus_y_ids[idx], self.corpus_y_mask[idx]




from sklearn.model_selection import train_test_split



def split_data(data, test_size=0.2, val_size=0.1):
    train_val, test = train_test_split(data, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size, random_state=42)
    return train, val, test


print("Using device:", device)
train, val, test = split_data(pretrain)
train_s, val_s, test_s = split_data(dfs[0])
train_i, val_i, test_i = split_data(dfs[1])
train_f, val_f, test_f = split_data(dfs[2])


pretrain_dataset = TranslationDataset(None, None, None, train, big_list_o_data, True, True, True)
preval_dataset = TranslationDataset(None, None, None, val, big_list_o_data, pretrain_dataset.token_vocab, False, True)
pretest_dataset = TranslationDataset(None, None, None, test, big_list_o_data, pretrain_dataset.token_vocab, False, True)


train_dataset = TranslationDataset(train_f, train_i, train_s, None, big_list_o_data, pretrain_dataset.token_vocab, False, False)
val_dataset = TranslationDataset(val_f, val_i, val_s, None, big_list_o_data, pretrain_dataset.token_vocab, False, False)
test_dataset = TranslationDataset(test_f, test_i, test_s, None, big_list_o_data, pretrain_dataset.token_vocab, False, False)

print(len(train_dataset.token_vocab))

x_ids, y_ids, _ = train_dataset[0]
decoded_x = [train_dataset.inverse_vocab[idx.item()] for idx in x_ids if idx.item() in train_dataset.inverse_vocab]
decoded_y = [train_dataset.inverse_vocab[idx.item()] for idx in y_ids if idx.item() in train_dataset.inverse_vocab]

print("Decoded x:", decoded_x)
print("Decoded y:", decoded_y)

x_ids, y_ids, pad_mask = train_dataset[0]


decoded_y = [
    train_dataset.inverse_vocab[idx.item()]
    for idx, mask in zip(y_ids, pad_mask)
    if mask.item() > 0
]


print("Decoded y (without padding):", decoded_y)
from torch.utils.data import DataLoader


#pretrain_train_loader = DataLoader(pretrain_dataset, batch_size=32, collate_fn=lambda batch: collate_fn(batch, pretrain_dataset))
#pretrain_val_loader = DataLoader(preval_dataset, batch_size=32, collate_fn=lambda batch: collate_fn(batch, preval_dataset))
#pretrain_test_loader = DataLoader(pretest_dataset, batch_size=32, collate_fn=lambda batch: collate_fn(batch, pretest_dataset))



from torch.utils.data import DataLoader


train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=lambda batch: collate_fn(batch, train_dataset))
"""val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=lambda batch: collate_fn(batch, val_dataset))
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=lambda batch: collate_fn(batch, test_dataset))"""


# this is to make it run on cpu for comet
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    collate_fn=lambda batch: collate_fn(batch, val_dataset),
    num_workers=0,
    pin_memory=False  #
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    collate_fn=lambda batch: collate_fn(batch, test_dataset),
    num_workers=0,
    pin_memory=False
)
from comet import download_model, load_from_checkpoint

device = torch.device("cpu")


comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)


comet_model.to(device)
comet_model.eval()

import torch
import torch.nn.functional as F

def collate_fn(batch, train_dataset):
    # Unzip the batch
    x_ids, y_ids, pad_masks = zip(*batch)
    #print("Pad masks before padding:")


    assert len(x_ids) == len(
        y_ids) == len(pad_masks), f"💡🚫   Error: x_ids and y_ids lengths do not match! len(x_ids): {len(x_ids)}, len(y_ids): {len(y_ids)}"


    max_len_x = max(x.size(0) for x in x_ids)
    #print("Calculated max_len x : ", max_len_x)
    max_len_pad = max(pad.size(0) for pad in pad_masks)
    #print("Calculated max_len pad: ", max_len_pad)
    max_len = max(max_len_pad, max_len_x)


    # Pad the x_ids and y_ids sequences to the max length
    x_batch = torch.stack(
        [torch.cat([torch.tensor(x), torch.full((max_len - len(x),), train_dataset.token_vocab["<PAD>"])]) for x in x_ids]
    ).long()

    y_batch = torch.stack(
        [torch.cat([torch.tensor(y), torch.full((max_len - len(y),), train_dataset.token_vocab["<PAD>"])]) for y in y_ids]
    ).long()

    # Create pad_mask, marking padding positions (True for padding)
    pad_batch = torch.stack(
        [torch.cat([torch.tensor(pad), torch.full((max_len - len(pad),), 0)]) for pad in pad_masks]
    ).long()

    # Convert pad_batch to boolean (True for padding, False for actual tokens)
    pad_batch = pad_batch.bool()



    return x_batch, y_batch, pad_batch


for x_batch, y_batch, pad_batch in train_loader:
    y_batch = y_batch.cpu()
    pad_batch = pad_batch.cpu()

    for batch_idx in range(min(2, y_batch.size(0))):
        y_ids = y_batch[batch_idx]
        pad_mask = pad_batch[batch_idx]


        decoded_y = [
            train_dataset.inverse_vocab[idx.item()]
            for idx, mask in zip(y_ids, pad_mask)
            if mask.item() > 0
        ]


        #print("Decoded y (without padding):", decoded_y)

for x_batch, y_batch, pad_batch in val_loader:
    y_batch = y_batch.cpu()
    pad_batch = pad_batch.cpu()

    for batch_idx in range(min(10, y_batch.size(0))):
        y_ids = y_batch[batch_idx]
        pad_mask = pad_batch[batch_idx]


        decoded_y = [
            val_dataset.inverse_vocab[idx.item()]
            for idx, mask in zip(y_ids, pad_mask)
            if mask.item() > 0
        ]


        #print("Decoded y for val (without padding):", decoded_y)
def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"💋👄👙💋👄👙💋👄👙NaN found in {name}!")



def find_max_sequence_length(dataset):
    max_length = max(len(x_ids) for x_ids in dataset.corpus_x_ids)
    print("min len: ", min(len(x_ids) for x_ids in dataset.corpus_x_ids))
    return max_length
print("Using device:", device)

pretrain_max_len = find_max_sequence_length(pretrain_dataset)
print("Max pretrain sequence length:", pretrain_max_len)

train_max_len = find_max_sequence_length(train_dataset)
print("Max train sequence length:", train_max_len)

test_max_len = find_max_sequence_length(test_dataset)
print("Max test sequence length:", test_max_len)

val_max_len = find_max_sequence_length(val_dataset)
print("Max val sequence length:", val_max_len)


max_seq_len = max(val_max_len, test_max_len, train_max_len)




#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, dropout=0.2)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = feedforward

    def forward(self, x, attn_mask):
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + feedforward_output)
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.mlp(x)
class DecoderTransformerTranslator(nn.Module):
    def __init__(self, max_seq_length, embed_dim, num_heads, vocab_size, train_dataset, attention_layers = 5):
        super(DecoderTransformerTranslator, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)  # will give us token embeddings
        self.position_embedding_table = nn.Embedding(max_seq_length, embed_dim) # we add some flexibility to the max sequence length in case there are longer sequences in test set
        self.last_linear_layer = nn.Linear(embed_dim,
                                 vocab_size)  # go from embeddings to outputs. Made sure that this is the vocab size in order to not have errors where the perplexity is artifically low
        # normalization for input

        self.norm1 = nn.LayerNorm(embed_dim)
        # normalization for attention output
        self.norm2 = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads

        self.train_dataset = train_dataset
        self.vocab_size = vocab_size
        self.feedforward = FeedForward(n_embd=embed_dim)  # will be applied after attention blocks

        self.attention_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, self.feedforward) for _ in range(attention_layers)]
        )





    def forward(self, x):
        assert torch.all(x >= 0) and torch.all(x < self.vocab_size), "Input indices are out of bounds!"
        check_nan(x, "input x")
        #print("x shape (batch size seq_len) ", x.shape)
        #print(type(x))
        #print("padding mask shape (batch size seq_len) ", x.shape)
        #print(type(x))
        batch_size, seq_length = x.shape
        token_embeddings = self.token_embedding_table(x)  # batch, seq len, embedding size

        check_nan(token_embeddings, "token embed")
        possitional_embeddings = self.position_embedding_table(torch.arange(seq_length, device=device))  # seq_len, embedding size
        #print("Token Embeddings shape (batch size, seq len, embed size): ", token_embeddings.shape)  # Should be [batch_size, seq_len, embedding_size]
        #print("Positional Embeddings: (seq len, embed size) ", possitional_embeddings.shape)
        check_nan(possitional_embeddings, "possitional_embeddings")
        #  I didnt realise this before, but each sentence will have there own positional embeddings based on size seq_len. For some reason I was under the impression earlier that each will have the same dim of block_size
        x = token_embeddings + possitional_embeddings  # element wise add. batch, seq_len, embedding size
        check_nan(x, "after adding embeddings")
        #print("Shape of x after addition (batch size, seq len, embed size) :", x.shape)
        #print(type(x))
        x = self.norm1(x)
        check_nan(x, "after norm1")
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1).bool()

        #print("mask shape: ", mask.shape)

        for layer in self.attention_layers:
            x = layer(x, attn_mask=mask)
            """

            # ^ the nn.nn.MultiheadAttention docs specify that in this function, the ones will be masked, the zeros wont be,
            # the mask should be of shape seq_len by seq_len because
            # each token will have Q and K where Q is of shape [seq_len x d], and K is of shape [seq_len x d] (where d doesnt really matter for our purposes)
            # after applying the dot product of Q * K^T,
            #  the final dimensions will be Q* K^T = [seq_len x d] * [d x seq_len ] = final dim:  [seq_len * seq_len]
            # the attention matrix will look something like this. Notice it is seq_len by seq_len
            # [dot_q1_k1, dot_q1_k2, ..., dot_q1_kn]
            # [dot_q2_k1, dot_q2_k2, ..., dot_q2_kn]
            # [dot_q3_k1, dot_q3_k2, ..., dot_q3_kn]
            # ...
            # [dot_qn_k1, dot_qm_k2, ..., dot_qn_kn]
            # This matrix contains the attention scores for each value, thus if we mask out the upper triangle, we are preventing
            # the key values ahead of the index we are on from combining with our query
            # In a multi-batch scenario, this process is repeated independently for each sequence in the batch,
            # so each sequence gets its own attention matrix.

            x, attn_weights = self.attention(x, x, x, attn_mask=mask) # used to initalize the key query and vector
            check_nan(x, "after attention")

            max_attention_weight = attn_weights.max()
            max_attention_idx = attn_weights.argmax()
            #print("arg max attention weight: ", max_attention_idx)
            x_norm = self.norm2(x)
            check_nan(x_norm, "after norm2")
            #print("shape x norm: ", x_norm.shape)
            #print("shape x: ", x.shape)
            fed_forward_x = self.feedforward(x_norm)
            check_nan(fed_forward_x, "after fed forward")
            #print("shape fedforward x: ", fed_forward_x.shape)
            #print("shape x + fedforward x: ", (x+ fed_forward_x).shape)
            x = x + fed_forward_x
            check_nan(x, "after resid connection")"""
        x = self.last_linear_layer(x)
        check_nan(x, "after linear")
        #print("x.shape)

        #print(x.shape)
        return x

import math
from nltk.translate.meteor_score import meteor_score
import numpy as np
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


model_name = "Babelscape/wikineural-multilingual-ner" # multilingual ner (same one used for preprocessing)
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")



def extract_entities(text):

    ner_results = nlp(text)

    noise_words = {
        "which", "what", "who", "why", "did", "w", "p", "whic", "of", "on the",
        "t", "o", "g", "the", "tri", "of the", "i",

        "quel", "quelle", "qui", "quoi", "pourquoi", "comment", "où", "combien",

        "cuál", "cual", "cu", "qué", "que", "quién", "quien", "cómo", "dónde", "por qué", "cuántos",

        "cuantos", "quale", "che", "chi", "cosa", "perché", "come", "dove", "quanto"
    }
    entities = [
        result["word"]
        for result in ner_results
        if result["word"].lower() not in noise_words
    ]

    print("ENTITIES DETECTED:", entities)
    return entities


def evaluate_epoch(model, data_loader, criterion, device, dataset, calculate=False):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_meteor_score = 0
    total_comet_score = 0
    total_batches = 0
    total_f1 = 0
    total_percision = 0
    total_recall = 0
    num_batches = 0

    with torch.no_grad():
        for x_batch, y_batch, pad_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).long()
            pad_batch = pad_batch.to(device).float()

            output = model(x_batch)

            output = output.view(-1, output.size(-1))
            y_batch = y_batch.view(-1)
            pad_batch = pad_batch.view(-1)

            loss = criterion(output, y_batch)

            masked_loss = loss * pad_batch
            total_loss += masked_loss.sum().item()
            total_tokens += pad_batch.sum().item()

            if calculate:
                max_values, max_indices = torch.max(output.view(-1, output.size(-1)), dim=1)
                decoded_sequences = []
                actual_sequences = []

                for batch_idx in range(x_batch.size(0)):
                    start_idx = batch_idx * x_batch.size(1)
                    end_idx = (batch_idx + 1) * x_batch.size(1)

                    pred_seq = [
                        dataset.inverse_vocab[idx.item()]
                        for idx, mask in zip(
                            max_indices[start_idx:end_idx],
                            pad_batch[start_idx:end_idx]
                        )
                        if mask.item() > 0
                    ]

                    true_seq = [
                        dataset.inverse_vocab[idx.item()]
                        for idx, mask in zip(
                            y_batch[start_idx:end_idx],
                            pad_batch[start_idx:end_idx]
                        )
                        if mask.item() > 0
                    ]

                    decoded_sequences.append(pred_seq)
                    actual_sequences.append(true_seq)

                pred_strs = [' '.join(seq).replace('▁', ' ') for seq in decoded_sequences]
                true_strs = [' '.join(seq).replace('▁', ' ') for seq in actual_sequences]
                print(pred_strs[0], true_strs[0]) # to see what is happening



                true_entities = [extract_entities(text) for text in true_strs]
                pred_entities = [extract_entities(text) for text in pred_strs]

                all_true = []
                all_pred = []

                for true, pred in zip(true_entities, pred_entities):
                    all_true.extend(true)
                    all_pred.extend(pred)


                true_set = set(all_true)
                pred_set = set(all_pred)

                tp = len(true_set & pred_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                total_f1 += f1
                total_percision += precision
                total_recall += recall
                num_batches += 1
                meteor_scores = [meteor_score([true.split()], pred.split()) for pred, true in zip(pred_strs, true_strs)]
                total_meteor_score += sum(meteor_scores)
                total_batches += len(meteor_scores)


                comet_inputs = [
                    {"src": src, "mt": pred, "ref": true}
                    for src, pred, true in zip(pred_strs, pred_strs, true_strs)
                ]
                comet_scores = comet_model.predict(comet_inputs, batch_size=8, gpus=0)["scores"]
                total_comet_score += sum(comet_scores)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if total_tokens > 0 else float('inf')

        avg_meteor = total_meteor_score / total_batches if total_batches > 0 else 0.0
        avg_comet = total_comet_score / total_batches if total_batches > 0 else 0.0
        avg_f1 = total_f1 / num_batches if num_batches > 0 else 0.0
        avg_precision = total_percision / num_batches if num_batches > 0 else 0.0
        avg_recall = total_recall / num_batches if num_batches > 0 else 0.0

        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation Perplexity: {perplexity:.4f}")
        print(f"Average METEOR score: {avg_meteor:.4f}")
        print(f"Average COMET score: {avg_comet:.4f}")
        print(f"Entity-level F1: {avg_f1:.4f}")
        print(f"Entity-level Precision: {avg_precision:.4f}")
        print(f"Entity-level Recall: {avg_recall:.4f}")

        return avg_loss, perplexity, avg_meteor, avg_comet, avg_f1, avg_precision, avg_recall




def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    for x_batch, y_batch, pad_batch in data_loader:

        # Move to the correct device (GPU/CPU)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pad_batch = pad_batch.to(device).float()


        optimizer.zero_grad()


        output = model(x_batch)

        output = output.view(-1, output.size(-1))
        y_batch = y_batch.view(-1)
        pad_batch = pad_batch.view(-1)

        loss = criterion(output, y_batch)

        masked_loss = loss * pad_batch
        total_loss += masked_loss.sum().item()
        total_tokens += pad_batch.sum().item()

        masked_loss.sum().backward()
        optimizer.step()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return avg_loss

# loading the model but adding new embeddings because the sequence length has changed from pretraining
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

# had to put this in for comet to work
device = torch.device("cpu")

model = DecoderTransformerTranslator(max_seq_len, 400, 5, len(train_dataset.token_vocab), train_dataset).to(device)
model = model.to(device)
# run the below code to go from pretrain to train
"""model_path = "/Users/darianlee/PycharmProjects/entity_preprocessing/______pretrain_model_more_data_from epoch 9 onwards_________.pth"

checkpoint = torch.load(model_path)

original_max_len = checkpoint['position_embedding_table.weight'].size(0)

new_max_len = max_seq_len
if new_max_len > original_max_len:
    # resizing the position embeddings to the new size (keeping the original weights, adding random for new positions)
    original_position_embeddings = checkpoint['position_embedding_table.weight']
    new_position_embeddings = torch.randn(new_max_len - original_max_len, original_position_embeddings.size(1),
                                          device=device)


    new_position_embeddings_combined = torch.cat((original_position_embeddings, new_position_embeddings), dim=0)


    checkpoint['position_embedding_table.weight'] = new_position_embeddings_combined


model.load_state_dict(checkpoint, strict=False)"""

model_path = "/Users/darianlee/PycharmProjects/entity_preprocessing/____finetuning_model_meteor________.pth"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
criterion = torch.nn.CrossEntropyLoss()
print("Using device:", device)
"""import torch
best_val_loss = float('inf')
best_meteor = 0
for epoch in range(50):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)



    val_loss, perplexity, meteor, comet = evaluate_epoch(model, val_loader, criterion, device, val_dataset, True)
    #test_loss, test_perplexity = evaluate_epoch(model, pretrain_test_loader, criterion, device, pretest_dataset, True)
    #print("🎄 Keep pushing, data elf! 🎁🎅✨Text perplexity is: ", test_perplexity)

    if meteor > best_meteor:

        best_meteor = meteor

        torch.save(model.state_dict(), '/Users/darianlee/PycharmProjects/entity_preprocessing/____finetuning_model_meteor________.pth')
        print(f"!!!!!🟥🟥🟥🟥🟥🟥🟥🟥 Model saved at epoch {epoch + 1} with meteor: {meteor}")
    print(f'🎄✨🎅 METEOR SCORE: {meteor}, COMET SCORE: {comet})
    print(f'🎄✨🎅 Keep spreading holiday cheer, Darian! You’re a shining star! 🌟🎁🎶')
    print(f'Validation Perplexity: {perplexity} 🎄❄️ Your hard work is the gift that keeps on giving – keep sleighing it! 🎁🎅✨')
    print('🎅✨ Keep believing, the season is full of magic and possibilities! ✨🎁🎄❄️\n')

    print(f'🌞🌺🌷✨ Keep blooming, Darian! You’re growing beautifully! 🌻🍃🌸')
    print(f'🌼 Every step you take is like a flower blooming! 🌸🌿 Epoch {epoch} – Keep your spirit high and your roots strong! 🌿🌻💪')
    print(f'Training Loss: {train_loss} 🌼 Keep nurturing your growth – you’re doing fantastic! 🌷🌺💫')
    print(f'Validation Loss: {val_loss} 🌻 You’re on the right path – every day you’re getting closer to your goal! 🌿🌸🌟')
    print('🌱💖 Keep blooming, the world is full of endless possibilities! 💖🌞🌿✨\n')"""





val_loss, perplexity, meteor, comet, f1, percision, recall = evaluate_epoch(model, test_loader, criterion, device, test_dataset, True)
# test_loss, test_perplexity = evaluate_epoch(model, pretrain_test_loader, criterion, device, pretest_dataset, True)
# print("🎄 Keep pushing, data elf! 🎁🎅✨Text perplexity is: ", test_perplexity)



print(f'🎄✨🎅 TESTING METEOR SCORE: {meteor}, TESTING COMET SCORE: {comet}')
print(f'🎄✨🎅 TESTING ENTITY LEVEL F1 SCORE: {f1}, TESTING ENTITY LEVEL PRECISION SCORE: {percision}, TESTING ENTITY LEVEL RECALL SCORE: {recall}')


val_loss, perplexity, meteor, comet, f1, percision, recall = evaluate_epoch(model, val_loader, criterion, device, val_dataset, True)
# test_loss, test_perplexity = evaluate_epoch(model, pretrain_test_loader, criterion, device, pretest_dataset, True)
# print("🎄 Keep pushing, data elf! 🎁🎅✨Text perplexity is: ", test_perplexity)



print(f'🎄✨🎅 VAL METEOR SCORE: {meteor}, VAL COMET SCORE: {comet}')
print(f'🎄✨🎅 VAL ENTITY LEVEL F1 SCORE: {f1}, VAL ENTITY LEVEL PRECISION SCORE: {percision}, VAL ENTITY LEVEL RECALL SCORE: {recall}')

