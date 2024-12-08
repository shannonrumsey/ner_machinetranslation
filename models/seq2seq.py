import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
pd.option_context('display.max_rows', None, 'display.max_columns', None)
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import sys
from evaluate import load
from pytorch_lightning import Trainer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Multilingual NER (same one used for preprocessing)
model_name = "Babelscape/wikineural-multilingual-ner" 
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

trainer = Trainer(accelerator="mps", devices=1)

seed = 27
torch.manual_seed(seed)

comet_metric = load("comet")
meteor_metric = load("meteor")

# Determine if CPU or GPU will be used
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.manual_seed(seed)
    print("Using MPS")

else:
    device = torch.device("cpu")
    print("Using CPU")

if device == torch.device("mps"):
    torch.mps.empty_cache()

es_data = os.path.join(os.path.dirname(__file__), "../processed_data/df_es.csv")
fr_data = os.path.join(os.path.dirname(__file__), "../processed_data/df_fr.csv")
it_data = os.path.join(os.path.dirname(__file__), "../processed_data/df_it.csv")

"""
Inputs: paths to 3 CSV files as a list
Assumes the CSV file is formatted as "../anything_{lang code}.csv"
Loads, formats, and combines the datasets
Outputs: x and y columns from combined dataframe
"""

def load_data(paths):
    df_lst = []
    for path in paths:
        old_df = pd.read_csv(path)
        source = old_df.apply(lambda row: f'<START> <{path.split("/")[-1].split("_")[-1].split(".")[0]}> {row["code_switch"]} <STOP>', axis=1)
        #target = old_df["target_no_punc"]
        target = old_df.apply(lambda row: f'<START> {row["target_no_punc"]} <STOP>', axis=1)
        df_lst.append(pd.DataFrame({"source": source, "target": target}))
    new_df = pd.concat(df_lst, ignore_index=True)
    return new_df["source"], new_df["target"]

x, y = load_data([es_data, fr_data, it_data])
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=27)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=27)

# Create vocabulary and translation dictionaries
class transDataset(Dataset):
    def __init__(self, x, y=None, token_vocab=None, trans_vocab=None, training=True, testing=False):
        # build dictionary that maps word/trans to a numeric value
        if training:
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1, "<START>": 2, "<STOP>": 3}
            self.trans_vocab = {'<PAD>': 0, "<UNK>": 1, "<START>": 2, "<STOP>": 3}

            for row in x:
                for token in row.split():
                    if token not in self.token_vocab:
                        # encode position when adding token into vocab (using length)
                        self.token_vocab[token] = len(self.token_vocab)
            for row in y:
                for token in row.split():
                    if token not in self.trans_vocab:
                        # encode position when adding trans into trans vocab (using length)
                        self.trans_vocab[token] = len(self.trans_vocab)
        else:
            assert token_vocab is not None and trans_vocab is not None
            self.token_vocab = token_vocab
            self.trans_vocab = trans_vocab
        
        # convert sentences and transs to numbers using the dictionary
        self.corpus_token_ids = []
        self.corpus_trans_ids = []

        # for prediction, a placeholder to determine where padding is
        if testing:
            for row in x:
                trans_ids = [1 for token in row.split()]
                self.corpus_trans_ids.append(torch.tensor(trans_ids))

            for row in x:
                token_ids = [self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in row.split()]
                self.corpus_token_ids.append(torch.tensor(token_ids))

        # for training AND validation
        if testing == False:
            for row in x:
                token_ids = [self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in row.split()]
                self.corpus_token_ids.append(torch.tensor(token_ids))

            for row in y:
                trans_ids = [self.trans_vocab.get(trans, self.trans_vocab["<UNK>"]) for trans in row.split()]
                self.corpus_trans_ids.append(torch.tensor(trans_ids))

    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        return self.corpus_token_ids[idx], self.corpus_trans_ids[idx]

"""
Inputs: batch of text that contains source and translation sequences
pads the input batch so that they match the length of the maximum sequence
outputs: padded source sentences and translations
"""
def padding(batch):
    token_ids = [item[0] for item in batch]
    trans_ids = [item[1] for item in batch]
    
    # set batch_first to True to make the batch size first dim
    padded_sentence = pad_sequence(token_ids, batch_first=True,
                                   padding_value=train_dataset.token_vocab["<PAD>"]).to(device)
    padded_trans = pad_sequence(trans_ids, batch_first=True,
                                padding_value=train_dataset.trans_vocab["<PAD>"]).to(device)
    return padded_sentence, padded_trans

# Example in x_train: I am cold -> [2, 4, 28]
# Example in y_train: J'ai froid -> [20, 53]
train_dataset = transDataset(x_train, y_train, training=True)
val_dataset = transDataset(x_val, y_val, token_vocab=train_dataset.token_vocab,
                           trans_vocab=train_dataset.trans_vocab, training=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=padding)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=padding)


# Encodes the meaning of the input sentence into a single vector
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
    
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        # hidden state of decoder
        self.Wa = nn.Linear(hidden_size, hidden_size)
        # hidden state of encoder
        self.Ua = nn.Linear(hidden_size, hidden_size)
        # combines projections into scalar attn score
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, queries, keys):
        # projects queries and keys and computes score for each encoder state
        scores = self.Va(torch.tanh(self.Wa(queries) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        # weighted sum of encoder outputs
        context = torch.bmm(weights, keys)

        return context, weights
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, encoder_hidden):
        batch_size = encoder_outputs.size(0)
        # tells the model which token index in the sequence to predict next
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long)
        # initialize decoder's hidden state with encoder's
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        all_attn_weights = []

        # decoding loop
        for i in range(100):
            # self.forward_step processes token and updates hidden state
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input,
                                                               decoder_hidden,
                                                               encoder_outputs)
            decoder_outputs.append(decoder_output)
            all_attn_weights.append(attn_weights)
            
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # probabilities of each token
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        all_attn_weights = torch.cat(all_attn_weights, dim=1)

        return decoder_outputs, decoder_hidden, all_attn_weights
    
    def forward_step(self, x, hidden, encoder_outputs):
        embeddings = self.embedding(x.to(device))
        output = self.dropout(embeddings)
        # reshape to make compatible batch size first dim
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embeddings, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

"""
Inputs: sentence from data
extracts the named enitities using spaCy in order to calculer NER F1 score
Outputs: named entities
"""
def get_named_entities(sentence):
    doc = nlp(sentence)
    noise_words = {"which", "what", "who", "why", "did", "w", "p", "whic", "of",
                   "on the", "t", "o", "g", "the", "tri", "of the", "i"}
    entities = [
        result["word"]
        for result in doc
        if result["word"].lower() not in noise_words
    ]
    return entities


# Training
input_size = len(train_dataset.token_vocab)
hidden_size = 512
output_size = len(train_dataset.trans_vocab)
num_layers = 1
dropout = 0.1
encoder = EncoderRNN(input_size, hidden_size, num_layers=num_layers, dropout=dropout).to(device)
decoder = AttnDecoderRNN(hidden_size, output_size, num_layers=num_layers, dropout=dropout).to(device)
pad_index = train_dataset.trans_vocab["<PAD>"]
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
encoder_optimizer = optim.AdamW(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.AdamW(decoder.parameters(), lr=0.001)
encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, factor=0.1)
decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, factor=0.1)

encoder_path = os.path.join(os.path.dirname(__file__), "trained_models/encoder_model")
decoder_path = os.path.join(os.path.dirname(__file__), "trained_models/decoder_model")

en_lookup = {idx: token for token, idx in train_dataset.token_vocab.items()}
trans_lookup = {idx: trans for trans, idx in train_dataset.trans_vocab.items()}

num_epoch = 15
prev_loss = None
for epoch in range(num_epoch):

    train_loss = 0
    encoder.train()
    decoder.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(x_batch.to(device))
        decoder_outputs, _, _ = decoder(encoder_outputs.to(device), encoder_hidden.to(device))

        # trim the decoder outputs so that it is compatible with y_batch
        decoder_outputs = decoder_outputs[:, :y_batch.shape[1], :]
        loss = loss_fn(decoder_outputs.reshape(-1,
                                               decoder_outputs.size(-1)),
                                               y_batch.view(-1))
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        train_loss += loss.item()

    print(f"Training loss on {epoch}: {train_loss / len(train_loader)}")

    encoder_scheduler.step(train_loss)
    decoder_scheduler.step(train_loss)

    print(f"Encoder LR: {encoder_scheduler.get_last_lr()[0]}, Decoder LR: {decoder_scheduler.get_last_lr()[0]}")

    encoder.eval()
    decoder.eval()
    val_loss = 0
    all_predictions = []
    all_trans = []
    all_sources = []
    f1_score = 0
    pred_entities = []
    actual_entities = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            encoder_outputs, encoder_hidden = encoder(x_val.to(device))
            decoder_outputs, _, _ = decoder(encoder_outputs.to(device), encoder_hidden.to(device))

            # Trim the decoder outputs so that it is compatible with y_batch
            decoder_outputs = decoder_outputs[:, :y_val.shape[1], :].contiguous()
            decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[-1])

            trans_ids = y_val.view(-1)
            source_ids = x_val.view(-1)
            loss = loss_fn(decoder_outputs, trans_ids)
            val_loss += loss.item()
            predictions = decoder_outputs.argmax(dim=-1).view(x_val.size(0), -1)

            ## Reformat data to calculate comet score

            # Machine translated predictions
            pred_count = 0
            for row in predictions:
                pred_count += 1
                sent_pred = []
                for token in row:
                    if token.item() not in {train_dataset.trans_vocab["<PAD>"], train_dataset.trans_vocab["<START>"]}:
                        if token.item() == train_dataset.trans_vocab["<STOP>"]:
                            break
                        sent_pred.append(trans_lookup[token.item()])
                all_predictions.append(" ".join(sent_pred))
                pred_entities.append(get_named_entities(" ".join(sent_pred)))


            # Actual translations
            trans_count = 0
            for row in y_val:
                trans_count += 1
                actual_trans = []
                for token in row:
                    if token.item() not in {train_dataset.trans_vocab["<PAD>"], train_dataset.trans_vocab["<START>"]}:
                        if token.item() == train_dataset.trans_vocab["<STOP>"]:
                            break
                        actual_trans.append(trans_lookup[token.item()])
                all_trans.append(" ".join(actual_trans))
                actual_entities.append(get_named_entities(" ".join(actual_trans)))

            # Source words in english
            for row in x_val:
                actual_sent = []
                for token in row:
                    if token.item() != train_dataset.token_vocab["<PAD>"]:
                        if token.item() == train_dataset.token_vocab["<STOP>"]:
                            actual_sent.append("<STOP>")
                            break
                        actual_sent.append(en_lookup[token.item()])
                all_sources.append(" ".join(actual_sent))
                
        # Prints the last sentence in the dataset
        print(f"Actual Sentence: {actual_trans}, Predicted Sentence: {sent_pred}")

        # Calculate F1 score
        actual_entities = set([entity for lst in actual_entities for entity in lst])
        pred_entities = set([entity for lst in pred_entities for entity in lst])
        # print(f"Actual entities: {actual_entities}, Pred: {pred_entities}")

        tp = len(actual_entities & pred_entities)
        fp = len(pred_entities - actual_entities)
        fn = len(actual_entities - pred_entities)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall)  if (precision + recall) > 0 else 0

        print(f"Val F1 Score on Epoch {epoch}: {f1}")

        total_val_loss = val_loss/ len(val_loader)
        print(f"Validation Loss on Epoch {epoch}: {total_val_loss}")

        comet_scores = comet_metric.compute(predictions=all_predictions, references=all_trans, sources=all_sources)
        print(f'Val COMET Score on Epoch {epoch}: {sum(comet_scores["scores"]) / len(comet_scores["scores"])}')

        meteor_scores = meteor_metric.compute(predictions=all_predictions, references=all_trans)
        print(f'Val METEOR Score on Epoch {epoch}: {meteor_scores["meteor"]}')

        # Save model with lowest loss
        if prev_loss is None or total_val_loss < prev_loss:
            print("LOWEST LOSS : SAVING MODEL")
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
            prev_loss = total_val_loss


# Run on test data
encoder = EncoderRNN(input_size, hidden_size, num_layers=num_layers, dropout=dropout).to(device)
decoder = AttnDecoderRNN(hidden_size, output_size, num_layers=num_layers, dropout=dropout).to(device)
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))
encoder.eval()
decoder.eval()
test_dataset = transDataset(x_test, y_test, token_vocab=train_dataset.token_vocab,
                           trans_vocab=train_dataset.trans_vocab, training=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=padding)

all_predictions = []
all_trans = []
all_sources = []
f1_score = 0
pred_entities = []
actual_entities = []

for x_test, y_test in test_loader:
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    encoder_outputs, encoder_hidden = encoder(x_test.to(device))
    decoder_outputs, _, _ = decoder(encoder_outputs.to(device), encoder_hidden.to(device))

    # Trim the decoder outputs so that it is compatible with y_batch
    decoder_outputs = decoder_outputs[:, :y_test.shape[1], :].contiguous()
    decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[-1])

    trans_ids = y_test.view(-1)
    source_ids = x_test.view(-1)
    loss = loss_fn(decoder_outputs, trans_ids)
    predictions = decoder_outputs.argmax(dim=-1).view(x_test.size(0), -1)

    ## Reformat data to calculate comet score

    # Machine translated predictions
    pred_count = 0
    for row in predictions:
        pred_count += 1
        sent_pred = []
        for token in row:
            if token.item() not in {train_dataset.trans_vocab["<PAD>"], train_dataset.trans_vocab["<START>"]}:
                if token.item() == train_dataset.trans_vocab["<STOP>"]:
                    break
                sent_pred.append(trans_lookup[token.item()])
        all_predictions.append(" ".join(sent_pred))
        pred_entities.append(get_named_entities(" ".join(sent_pred)))


    # Actual translations
    trans_count = 0
    for row in y_test:
        trans_count += 1
        actual_trans = []
        for token in row:
            if token.item() not in {train_dataset.trans_vocab["<PAD>"], train_dataset.trans_vocab["<START>"]}:
                if token.item() == train_dataset.trans_vocab["<STOP>"]:
                    break
                actual_trans.append(trans_lookup[token.item()])
        all_trans.append(" ".join(actual_trans))
        actual_entities.append(get_named_entities(" ".join(actual_trans)))

    # Source words in english
    for row in x_test:
        actual_sent = []
        for token in row:
            if token.item() != train_dataset.token_vocab["<PAD>"]:
                if token.item() == train_dataset.token_vocab["<STOP>"]:
                    actual_sent.append("<STOP>")
                    break
                actual_sent.append(en_lookup[token.item()])
        all_sources.append(" ".join(actual_sent))
        
# Prints the last sentence in the dataset
print(f"Actual Sentence: {actual_trans}, Predicted Sentence: {sent_pred}")

# Calculate F1 score
actual_entities = set([entity for lst in actual_entities for entity in lst])
pred_entities = set([entity for lst in pred_entities for entity in lst])
# print(f"Actual entities: {actual_entities}, Pred: {pred_entities}")

tp = len(actual_entities & pred_entities)
fp = len(pred_entities - actual_entities)
fn = len(actual_entities - pred_entities)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall)  if (precision + recall) > 0 else 0

print(f"Test F1 Score: {f1}")

comet_scores = comet_metric.compute(predictions=all_predictions, references=all_trans, sources=all_sources)
print(f'Test COMET Score: {sum(comet_scores["scores"]) / len(comet_scores["scores"])}')

meteor_scores = meteor_metric.compute(predictions=all_predictions, references=all_trans)
print(f'Test METEOR Score: {meteor_scores["meteor"]}')

