# OCR LLM - Summer 2024 OCR Correction Project

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
#using ocrenv

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
- **Script:** `writecsv.py` (extracts data, cleans, reformats for LLM training)
- **Script:** `read_data.py` (combines OCR and original data, cleans up)

### Tokenize the Text
- **Script:** `tokenize_data.py` (labels tokens for training)

### Training
- **Script:** `train_bart.py` (basic training script, runs on two GPUs)
- **Template:** `train_bart_template.sh` (basic template for the Slurm script)
- **Script:** `generate_training_scripts` (creates training scripts for different models)
- **Script:** `submit_bart_training` (submits all models for training)

### Plot Training Results
- **Script:** `plot_bart_training` (plots training results) *#untested*

## Testing
- **Script:** `bart_test.py` (generates results)
- **Script:** `bart_test_validate.py` (uses validation tools)

### Process The OCR
- **Script:** `run_bart_slurm.sh` (modifiable for more GPUs, processes OCR text)
- **Script:** `/data/lhyman6/OCR/scripts/bart_over_data_modeltest_debug_1.py` (tests current model, checks model list)

---

## LLAVA Training
#using llavaenv
# need to build flash-attn and everything while on cuda gpu

### Data Preparation
- **Script:** `llava_data_read.py` (links images and texts into JSON format for LLAVA training, uses `complete_bart_training_data.csv` from BART training)
output folder: /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/

### Model Training
- **Script:** `train_mem.py` (from [LLaVA GitHub](https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train_mem.py))

finetune instructions: https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md

make sure you have 1) cuda 2) path deepspeed 3) ./fine_tune_lora.sh
#check your paths
#i altered the training code to push to gpu
#needs cutlass, needs torch 1.9.1
torch 2.1.2

HOWTO
fine_tune_lora.sh is the basic template which runs train.py and zero3.json (the config file)
generate_llava_scripts.py creates the versions of that for each type of training


need to find the train.py that is deep in the files and make sure that it is pushed to github.