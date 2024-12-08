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
