### Step 1: Run `creating_lookup_table.py`

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


