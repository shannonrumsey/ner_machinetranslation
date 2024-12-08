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
### Step 1: Run `creating_lookup_table.py`

First, execute `creating_lookup_table.py` to generate the necessary JSON files for use in `code_switching.py`.

**Purpose**:  
`creating_lookup_table.py` creates a dictionary lookup table for the entities in the SemEval dataset.

**Command to Run**:
```bash
python creating_lookup_table.py \
  --data_es path/to/spanish_data.jsonl \
  --data_it path/to/italian_data.jsonl \
  --data_fr path/to/french_data.jsonl \
  --output_es path/to/save/spanish_lookup.json \
  --output_it path/to/save/italian_lookup.json \
  --output_fr path/to/save/french_lookup.json
```

### Note:
The original JSON files can be obtained from the SemEval website.

### Dependencies:
All required dependencies can be found in the `requirements.txt` file.

---

### Step 2: Run `code_switching.py`

After generating the necessary lookup tables, you can use `code_switching.py` to process the data and create a CSV with code-switched entities.

**Purpose**:  
`code_switching.py` generates a CSV file with code-switched entities from the provided data and lookup tables.

**Command to Run**:
```bash
python code_switching.py \
  --lookup_fr path_to_french_lookup_table \
  --lookup_it path_to_italian_lookup_table \
  --lookup_es path_to_spanish_lookup_table \
  --data_fr path_to_french_data \
  --data_it path_to_italian_data \
  --data_es path_to_spanish_data \
  --output_fr path_to_save_french_output \
  --output_it path_to_save_italian_output \
  --output_es path_to_save_spanish_output
```



### Note:
The original JSON files can be obtained from the SemEval website.

### Dependencies:
All required dependencies can be found in the `requirements.txt` file.


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
   

