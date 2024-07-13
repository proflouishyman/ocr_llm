# OCR LLM - Summer 2024 OCR Correction Project

## Original Image Files
- **Path:**      `/data/lhyman6/OCR/data/images/*.jpg`

## Pytesseract Image Files
- **Path:**      `/data/lhyman6/OCR/data/images/*.txt.pyte`

## Chat-GPT Vision Files
- **Path:**      `/data/lhyman6/OCR/data/images/*.txt`

---

## Experiment Design
The project aims to compare the training of LLMs for OCR with different levels of gold and silver quality data.

### Models and Data Sizes
- **BART**        (100, 1000, 10000)
- **LLAVA**       (untuned, 100, 1000, 10000)

### Nomenclature
- Example: `Bart, 100, Gold` = `bart_100_gold`

---

## Batch Operations
Using `ocrenv`:

- **Batch Upload:**     Send all the images to Chat 4o for processing.
- **Batch Download:**   Retrieve all the data.
- **Write CSV:**        Read the OCR text into a file called `silver_ocr_data.csv`.

---

## Bart Training
Train BART to correct OCR with both gold and silver data in increments of 100, 1000, and 10000.
- **Gold Data:**    `/data/lhyman6/OCR/data/gompers_corrections/bythepeople1.csv`
- **Silver Data:**  `/data/lhyman6/OCR/scripts/ocr_llm/silver_ocr_data.csv`

---

## BART Training Workflow

### Download Images
- **Script:**      `download_images.py`                (this needs to be modified for the second CSV file)

### OCR Images with Slurm
- **Scripts:**     `ocr_pyte.py` and `ocr_slurm.sh`    (processes images into Pytesseract)

### Preprocess the OCR Text
- **Script:**      `writecsv.py`                       (extracts data, cleans, reformats for LLM training)
- **Script:**      `read_data.py`                      (combines OCR and original data, cleans up)

### Tokenize the Text
- **Script:**      `tokenize_data.py`                  (labels tokens for training)

### Training
- **Script:**      `train_bart.py`                     (basic training script, runs on two GPUs)
- **Template:**    `train_bart_template.sh`            (basic template for the Slurm script)
- **Script:**      `generate_training_scripts`         (creates training scripts for different models)
- **Script:**      `submit_bart_training`              (submits all models for training)

### Plot Training Results
- **Script:**      `plot_bart_training`                (plots training results) *#untested*

---

## Testing
- **Script:**      `bart_test.py`                      (generates results)
- **Script:**      `bart_test_validate.py`             (uses validation tools)

### Process The OCR
- **Script:**      `run_bart_slurm.sh`                 (modifiable for more GPUs, processes OCR text)
- **Script:**      `/data/lhyman6/OCR/scripts/bart_over_data_modeltest_debug_1.py`  (tests current model, checks model list)

---

## LLAVA Training
Using `llavaenv`, need to build flash-attn and everything while on cuda gpu.

### Data Preparation
- **Script:**      `llava_data_read.py`                (links images and texts into JSON format for LLAVA training, uses `complete_bart_training_data.csv` from BART training)
- **Output Folder:** `/scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/`

### Model Training
- **Script:**      `train_mem.py`                      (from [LLaVA GitHub](https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train_mem.py))

**Finetune Instructions:** [LLaVA Finetune Documentation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)

Make sure you have:
1. `cuda/12.1.0`
2. Paths to `deepspeed` and path to `cutlass`

`train.py` is backed up locally by `move_train.py` but the version you need to edit in order to make changes is actually `"/home/lhyman6/.local/lib/python3.8/site-packages/llava/train/train.py"`

### HOWTO
- **Script:**      `fine_tune_lora.sh`                  (basic template which runs `train.py` and `zero3.json` (the config file))
- **Script:**      `generate_llava_scripts.py`          (creates the versions of that for each type of training)
- **Script:**      `submit_llava_METAL_NUMBER.sh`       (submit these with `sbatch` to run training)

---

## ANALYSIS

Now that we have models, we need to test the results. We need to construct a CSV file that looks like this:

| id | transcription | pyte_ocr | chatgpt_ocr | BART_untuned | BART_gold_100 | BART_gold_1000 | BART_gold_10000 | BART_silver_100 | BART_silver_1000 | BART_silver_10000 | LLAVA_untuned | LLAVA_gold_100 | LLAVA_gold_1000 | LLAVA_gold_10000 | LLAVA_silver_100 | LLAVA_silver_1000 | LLAVA_silver_10000 |
|----|---------------|----------|-------------|--------------|---------------|----------------|-----------------|-----------------|------------------|-------------------|---------------|----------------|-----------------|------------------|------------------|-------------------|--------------------|
|    |               |          |             |              |               |                |                 |                 |                  |                   |               |                |                 |                  |                  |                   |                    |
|    |               |          |             |              |               |                |                 |                 |                  |                   |               |                |                 |                  |                  |                   |                    |

- **Script:**   `download_images.py`                (this needs to be modified for the second CSV file)
- **Scripts:**  `ocr_pyte.py` and `ocr_slurm.sh`    (processes images into Pytesseract)

- **Scripts:**  `generate_complete_testing.py`     this script reads in the LOC CSV and then reads in additional data (pyte OCR and openai OCR) from the data directory
creates complete_testing_csv.csv

- **Scripts:**  'run_bart_models.py"    runs the trained models over the data
creates processed_testing_csv.csv
