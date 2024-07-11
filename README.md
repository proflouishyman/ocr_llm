# ocr_llm
Summer 2024 ocr correction project



Read me for OCR LLM


Original Image Files:
/data/lhyman6/OCR/data/images

Pytesseract Image Files:
/scratch4/lhyman6/1919/1919/images_pyte

Chat-GPT Vision Files:
[TBD] [Need to be batch uploaded]




---
Experiment Design
Compare the training of LLMs for OCR with different levels of gold and silver quality data.

BART (100, 1000, 10000)
LLAVA (100, 1000, 10000)

The nomenclature is:
Bart, 100, Gold = bart_100_gold


---
TO DO
0. Batch submit sample to chatgpt. in the process extract information. doublecheck this hasnt been done before DONE
1. Clean and organize the samples so that there are no handwriting examples DONE
2. Double check consistent names with the previous OCR pytesseract data DONE
3. Setup concurrent training systems for the different models with exhaustion epoch limits
4. Check results


---

batch upload: send all the images to chat 4o for processing
batch download: retrieve all the data
writecsv: read the ocr text into a file called silver_ocr_data.csv

Bart Training:
Train BART to correct OCR with both gold and silver data in increments of 100, 1000, and 10000
gold data: /data/lhyman6/OCR/data/gompers_corrections/bythepeople1.csv
silver data: /data/lhyman6/OCR/scripts/ocr_llm/silver_ocr_data.csv



----
ORIGINAL BART TRAINING 

Download Images

/data/lhyman6/OCR/scripts/download_images.py

OCR Images with Slurm
/data/lhyman6/OCR/scripts/ocr_slurm.sh
Uses ocr_pyte.py

Preprocess the OCR Text
read_data.py                combines the ocr and original data, cleans up

Tokenize the Text
tokenize_data.py            turns the text into labelled tokens for training 

Training
train_bart.py               the basic training script. Runs on two GPUS
train_bart_template.sh      the basic template for the slurm script
generate_training_scripts   generates the training scripts for the different models
submit_bart_training        submits all the models



plot_bart_training         plots the results of the training #untested

Testing
bart_test.py            #generates results
bart_test_validate.py   #uses validation tools

Process The OCR
run_bart_slurm.sh       #can be modified for many more GPUS but processes the OCR text


------ Running the data against the existing letters
OCR the images
/data/lhyman6/OCR/scripts/ocr_slurm.sh #current version does not check for existence.

Load results into csv file
/data/lhyman6/OCR/scripts/load_ocr_data.py #now checks for strings

Process The OCR
run_bart_slurm.sh     #currently set to /scratch4/lhyman6/OCR/work/tuning_results_robust/checkpoint-26440
/data/lhyman6/OCR/scripts/bart_over_data_modeltest_debug_1.py #current model. check for model list

---TESTING MODELS
/data/lhyman6/OCR/scripts/bart_over_data_modeltest_debug_1.py #runs a test batch for every model to compare results