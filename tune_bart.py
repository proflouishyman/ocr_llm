from transformers import Trainer, TrainingArguments, BartForConditionalGeneration, EarlyStoppingCallback
from datasets import load_from_disk
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# Boolean flag to control local or distributed execution
run_locally = True

# Function to setup distributed training
def setup_distributed_training():
    if not run_locally:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

# Function to cleanup distributed training
def cleanup_distributed_training():
    if not run_locally:
        dist.destroy_process_group()

# List and confirm the setup of available GPUs
print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Load datasets
train_dataset = load_from_disk('/scratch4/lhyman6/OCR/work/train_dataset')
validation_dataset = load_from_disk('/scratch4/lhyman6/OCR/work/validation_dataset')
test_dataset = load_from_disk('/scratch4/lhyman6/OCR/work/test_dataset')

print("Loading Model")
# Load model
model = BartForConditionalGeneration.from_pretrained('/scratch4/lhyman6/OCR/work/tuning_results_robust/checkpoint-26440')  # Replace 'XXXXX' with the checkpoint number

device = setup_distributed_training()
model.to(device)

if not run_locally:
    # If using DDP, wrap model
    model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank())

# Set up training arguments
training_args = TrainingArguments(
    output_dir='/scratch4/lhyman6/OCR/work/tuning_results_robust',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Reduce batch size to fit in memory
    per_device_eval_batch_size=4,   # Reduce batch size to fit in memory
    gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps to simulate larger batch size
    num_train_epochs=100,  # Allow for potentially long training
    weight_decay=0.02,
    save_strategy="epoch",
    save_total_limit=5,
    report_to="none",
    dataloader_drop_last=True,
    lr_scheduler_type='linear',
    warmup_ratio=0.1,
    resume_from_checkpoint=True,  # Set to True to resume training from the checkpoint
    load_best_model_at_end=True,  # Required for early stopping
    fp16=True  # Use mixed precision training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add early stopping callback
)

print("Start the training!")
# Start training
trainer.train()

# Clean up the process group if running in distributed mode
cleanup_distributed_training()
