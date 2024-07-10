from datasets import load_from_disk

# Load the full tokenized dataset
full_dataset = load_from_disk('/scratch4/lhyman6/OCR/work/tokenized_datasets')

# Split the dataset into training and testing sets
train_test_split = full_dataset.train_test_split(test_size=0.1)  # 10% for testing, 90% for training

# Assign the training and test datasets
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Further split the training set to create a validation set
train_validation_split = train_dataset.train_test_split(test_size=0.1)  # 10% of the training set for validation

# Finalize the training and validation datasets
train_dataset = train_validation_split['train']
validation_dataset = train_validation_split['test']

# Optionally, save the splits to disk for easy reloading
train_dataset.save_to_disk('/scratch4/lhyman6/OCR/work/train_dataset')
validation_dataset.save_to_disk('/scratch4/lhyman6/OCR/work/validation_dataset')
test_dataset.save_to_disk('/scratch4/lhyman6/OCR/work/test_dataset')



print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(validation_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")



# Load datasets
train_dataset = load_from_disk('/scratch4/lhyman6/OCR/work/train_dataset')
validation_dataset = load_from_disk('/scratch4/lhyman6/OCR/work/validation_dataset')
test_dataset = load_from_disk('/scratch4/lhyman6/OCR/work/test_dataset')

# Sample check or review
#print(train_dataset[0])  # Print the first example from the training set
#print(validation_dataset[0])  # Print the first example from the validation set
#print(test_dataset[0])  # Print the first example from the test set