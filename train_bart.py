from transformers import Trainer, TrainingArguments, BartForConditionalGeneration, EarlyStoppingCallback
from datasets import load_from_disk
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train BART model")
parser.add_argument('--run_locally', action='store_true', help="Flag to control local or distributed execution")
parser.add_argument('--train_dataset', type=str, required=True, help="Path to the training dataset")
parser.add_argument('--validation_dataset', type=str, required=True, help="Path to the validation dataset")
parser.add_argument('--test_dataset', type=str, required=True, help="Path to the test dataset")
parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the model and outputs")
parser.add_argument('--checkpoint', type=str, help="Checkpoint path to resume training")
args = parser.parse_args()

# Boolean flag to control local or distributed execution
run_locally = args.run_locally

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
train_dataset = load_from_disk(args.train_dataset)
validation_dataset = load_from_disk(args.validation_dataset)
test_dataset = load_from_disk(args.test_dataset)

print("Loading Model")
# Load model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

device = setup_distributed_training()
model.to(device)

if not run_locally:
    # If using DDP, wrap model
    model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank())

# Set up training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    logging_dir=os.path.join(args.output_dir, 'logs'),
    logging_first_step=True,
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Reduce batch size to fit in memory
    per_device_eval_batch_size=4,   # Reduce batch size to fit in memory
    gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps to simulate larger batch size
    num_train_epochs=100,  # Allow for potentially long training
    weight_decay=0.02,
    save_strategy="epoch",
    save_total_limit=5,
    report_to="tensorboard",
    dataloader_drop_last=True,
    lr_scheduler_type='linear',
    warmup_ratio=0.1,
    resume_from_checkpoint=args.checkpoint is not None,  # Resume from checkpoint if provided
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
if args.checkpoint:
    trainer.train(resume_from_checkpoint=args.checkpoint)
else:
    trainer.train()

# Clean up the process group if running in distributed mode
cleanup_distributed_training()
