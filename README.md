# OCR LLM
**Summer 2024 OCR Correction Project**

## Original Image Files
- Path: `/data/lhyman6/OCR/data/images/*.jpg`

## Pytesseract Image Files
- Path: `/data/lhyman6/OCR/data/images/*.txt.pyte`

## Chat-GPT Vision Files
- Path: `/data/lhyman6/OCR/data/images/*.txt`

---

## Experiment Design
The project aims to compare the training of LLMs for OCR with different levels of gold and silver quality data.

### Models and Data Sizes
- **BART** (100, 1000, 10000)
- **LLAVA** (100, 1000, 10000)

### Nomenclature
- Example: `Bart, 100, Gold` = `bart_100_gold`

---

## Batch Operations
- **Batch Upload:** Send all the images to Chat 4o for processing.
- **Batch Download:** Retrieve all the data.
- **Write CSV:** Read the OCR text into a file called `silver_ocr_data.csv`.

## Bart Training
Train BART to correct OCR with both gold and silver data in increments of 100, 1000, and 10000.
- **Gold Data:** `/data/lhyman6/OCR/data/gompers_corrections/bythepeople1.csv`
- **Silver Data:** `/data/lhyman6/OCR/scripts/ocr_llm/silver_ocr_data.csv`

---

## BART Training Workflow

### Download Images
- Script: `/data/lhyman6/OCR/scripts/download_images.py`

### OCR Images with Slurm
- **Script:** `ocr_slurm.sh`
- **Script:** `ocr_pyte.py` (uses Pytesseract to image all the files)

### Preprocess the OCR Text
- **Script:** `read_data.py` (combines the OCR and original data, cleans up)

### Tokenize the Text
- **Script:** `tokenize_data.py` (turns the text into labelled tokens for training)

### Training
- **Script:** `train_bart.py` (basic training script, runs on two GPUs)
- **Template:** `train_bart_template.sh` (basic template for the Slurm script)
- **Script:** `generate_training_scripts` (generates the training scripts for different models)
- **Script:** `submit_bart_training` (submits all the models)

### Plot Training Results
- **Script:** `plot_bart_training` (plots the results of the training) *#untested*

## Testing
- **Script:** `bart_test.py` (generates results)
- **Script:** `bart_test_validate.py` (uses validation tools)

### Process The OCR
- **Script:** `run_bart_slurm.sh` (can be modified for many more GPUs but processes the OCR text)
- **Script:** `run_bart_slurm.sh` (currently set to `/scratch4/lhyman6/OCR/work/tuning_results_robust/checkpoint-26440`)
- **Script:** `/data/lhyman6/OCR/scripts/bart_over_data_modeltest_debug_1.py` (current model, check for model list)

## Testing Models
- **Script:** `/data/lhyman6/OCR/scripts/bart_over_data_modeltest_debug_1.py` (runs a test batch for every model to compare results)
