# Entity Aware Machine Translation

# Repo Structure
```plaintext
├── models
    ├── seq2seq.py
└── semeval
    ├── sample/
    ├── train/
└── Final data preprocessing and codeswitching (with runnable scripts)
    ├── code_switching copy.py/
    ├── creating_lookup_table copy.py/
    ├── HOW_TO_RUN.md/
    ├── requirements.txt/
└── Pretraining and decoder only final model
    ├── pretraining.py/
    ├── translation_transformer.py/
    ├── HOW_TO_RUN.md/
├── Links to final models.md/
├── processed_data/
└──README.md
```
The semeval folder is the original source of the data. It is manipulated to formulate the actual datasets used in the models (under the preprocessed_data folder). 
## Seq2Seq model
- Encoder-Decoder GRU model + Attention
- Install requirements
```bash
pip install -r models/seq_requirements.txt
```
- To run model file:
```bash
python models/seq2seq.py
```
- running the file using the above command will use the data from the repository automatically.

## Transformer model
- 
- ### Step 1: Get preprocessed data: Run `creating_lookup_table.py`

First, execute `pretraining.py` to generate the necessary blocks.csv for use in `translation_transformer.py`.

**Purpose**:  
`pretraining.py` creates a csv file which will be used as input for the pretraining 

**Command to Run**:
```bash
python pretraining.py \
  path/to/english_squad.jsonl \
  path/to/french_data.csv \
  path/to/italian_data.csv \
  path/to/spanish_data.csv \
  path/to/save/csv_files/
```


### Note:
 The code will upload the french, italian, and spanih SQuAD from the web, however the English SQuAD must be provided. The English SQuAD can be found on the SQuAD website. The user must also input the csv files they obtained in the code_switching.py file

### Dependencies:
All required dependencies can be found in the `requirements.txt` file.

---

### Step 2: Run `code_switching.py`

After generating the necessary lookup tables, you can use `code_switching.py` to process the data and create a CSV with code-switched entities.

**Purpose**:  
`translation_transformer.py` pretrains, trains and evaluates the model

### IMPORTANT NOTE ABOUT RUNNING:
The code provided has been commeted out to only run the evaluation on the test set using the best model. To train the model, uncomment out the training loop. To pretrain the model, uncommet out the part of the file related to pretraining.
We plan to seperate these into seperate files for a more streamlined user experience in the near future. 
**Command to Run**:
```bash
python translation_transformer.py \
  path/to/folder_containing_csv_files/ \
  path/to/pretrain_csv_file.csv \
  path/to/your_finetuned_model.pth
```

### Step 3: Get pretrain data: 

First, execute `pretraining.py` to generate the necessary blocks.csv for use in `translation_transformer.py`.

**Purpose**:  
`pretraining.py` creates a csv file which will be used as input for the pretraining 

**Command to Run**:
```bash
python pretraining.py \
  path/to/english_squad.jsonl \
  path/to/french_data.csv \
  path/to/italian_data.csv \
  path/to/spanish_data.csv \
  path/to/save/csv_files/
```


### Note:
 The code will upload the french, italian, and spanih SQuAD from the web, however the English SQuAD must be provided. The English SQuAD can be found on the SQuAD website. The user must also input the csv files they obtained in the code_switching.py file

### Dependencies:
All required dependencies can be found in the `requirements.txt` file.

---

### Step 4: Run `translation_transformer.py` to pretrain, train, and test the final model

After generating the necessary lookup tables, you can use `code_switching.py` to process the data and create a CSV with code-switched entities.

**Purpose**:  
`translation_transformer.py` pretrains, trains and evaluates the model

### IMPORTANT NOTE ABOUT RUNNING:
The code provided has been commented out to only run the evaluation on the test set using the best model. To train the model, uncomment out the training loop. To pretrain the model, uncomment out the part of the file related to pretraining.
We plan to seperate these into seperate files for a more streamlined user experience in the near future. To find our pretrained and final models, see Links to final models.md in the github.
**Command to Run**:
```bash
python translation_transformer.py \
  path/to/folder_containing_csv_files/ \
  path/to/pretrain_csv_file.csv \
  path/to/your_finetuned_model.pth
   

