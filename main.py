from tqdm import tqdm
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
import json
import math
import multiprocessing
from torch.cuda.amp import autocast, GradScaler  # Import for mixed precision
from generate_config import get_adapter_mapping

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.

# For deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def split_dataset_by_court(dataset):
    court_datasets = {}
    courts = set(dataset['court'])
    for court in courts:
        # Filter dataset for this court
        court_dataset = dataset.filter(lambda example: example['court'] == court)
        court_datasets[court] = court_dataset
    return court_datasets

def create_dataloaders(court_datasets, batch_size, shuffle=True):
    dataloaders = {}
    for court, dataset in court_datasets.items():
        # Ensure the dataset size is divisible by batch_size
        num_samples = len(dataset)
        num_full_batches = num_samples // batch_size
        num_samples_to_use = num_full_batches * batch_size
        dataset = dataset.select(range(num_samples_to_use))

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=True)
        dataloaders[court] = dataloader
    return dataloaders


import itertools

def get_combined_dataloader(dataloaders):
    all_batches = []
    for dataloader in dataloaders.values():
        for batch in dataloader:
            all_batches.append(batch)
    # Shuffle all batches
    random.shuffle(all_batches)
    return all_batches


# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-xl')
model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-xl', attn_implementation="flash_attention_2")

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

# Load adapter configurations from 'config.json'
with open('config.json', 'r') as f:
    loaded_configs = json.load(f)

# Access the configurations
adapter_configs = loaded_configs['adapter_configs']

def create_lora_config(adapter_cfg):
    return LoraConfig(
        r=adapter_cfg['r'],
        lora_alpha=adapter_cfg['lora_alpha'],
        lora_dropout=adapter_cfg['lora_dropout'],
        target_modules=['c_attn', 'c_proj'],
        layers_to_transform=adapter_cfg['layers_to_transform'],
        layers_pattern='h',
        bias='none',
        task_type='CAUSAL_LM'
    )

# Initialize the PEFT model with the first adapter
first_adapter = adapter_configs[0]
peft_config = create_lora_config(first_adapter)
model = get_peft_model(model, peft_config, adapter_name=first_adapter['adapter_name'])

# Add the remaining adapters
for adapter_cfg in adapter_configs[1:]:
    peft_config = create_lora_config(adapter_cfg)
    model.add_adapter(adapter_cfg['adapter_name'], peft_config)

# Create a mapping from dataset_name to court code
dataset_name_to_court = {
    'santoshtyss/uk_courts_cases': 'UKC',
    'santoshtyss/eu-court-cases': 'EUC',
    'santoshtyss/indian_courts_cases': 'IC',
    'santoshtyss/ecthr_cases': 'ECHR',
    'santoshtyss/canadian_court_cases': 'CAC'
}

# Load your dataset
dataset = load_dataset('MHGanainy/multi_clustering', 'lex-former-8-clustered-instance-b-dataset-cluster')

block_size = 1024

# Define the tokenize function
def tokenize_function(examples):
    # Tokenize the text
    result = tokenizer(
        examples["original_text"],
        padding='max_length',
        max_length=block_size,
        truncation=True
    )
    # Convert input_ids to NumPy array
    input_ids = np.array(result["input_ids"])
    # Copy input_ids to labels
    labels = input_ids.copy()
    # Set the first 512 tokens of labels to -100
    labels[:, :512] = -100
    # Set labels to -100 where input_ids == pad_token_id
    labels[input_ids == tokenizer.pad_token_id] = -100
    # Convert back to lists
    result["labels"] = labels.tolist()

    # Map dataset_name to court code
    courts = [dataset_name_to_court.get(name, 'Unknown') for name in examples['dataset_name']]
    if 'Unknown' in courts:
        raise ValueError("Found unknown dataset_name in examples.")

    result['court'] = courts  # Keep court code in the result

    # Generate adapter names based on the court field
    adapter_names = []
    for court in courts:
        adapter_mapping = get_adapter_mapping(court)
        # Convert layer indices to strings for consistency
        adapter_names.append({str(k): v for k, v in adapter_mapping.items()})
    result['adapter_names'] = adapter_names

    # Exclude 'original_text' and 'dataset_name' from the result
    return result


def prepare_dataset(dataset_split, split="train"):
    total_cores = multiprocessing.cpu_count()
    num_cpu_cores = min(64, total_cores)
    print(f"Using {num_cpu_cores} CPU cores for '{split}' dataset processing.")

    lm_dataset = dataset_split.map(
        tokenize_function,
        batched=True,
        remove_columns=['original_text', 'cluster_id', 'dataset_name'],  # Keep 'court'
        desc=f"Tokenizing {split} dataset",
        num_proc=num_cpu_cores,
    )

    lm_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)
    return lm_dataset


print("Preprocessing training data...")
train_dataset = prepare_dataset(dataset["train"].select(range(0,10)), "train")

print("Preprocessing validation data...")
eval_dataset = prepare_dataset(dataset["validation"].select(range(0,10)), "validation")

# Define the data collator
class DataCollatorWithAdapterNames(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Extract adapter_names and remove from features
        adapter_names = [feature.pop('adapter_names') for feature in features]
        # Remove 'court' from features
        for feature in features:
            feature.pop('court', None)  # Remove 'court' if present
    
        # Use the parent class method to collate input_ids and labels
        batch = super().__call__(features)
    
        # Add adapter_names back to the batch
        batch['adapter_names'] = adapter_names
    
        return batch


data_collator = DataCollatorWithAdapterNames(tokenizer=tokenizer, mlm=False)


print("Splitting training data by court...")
train_court_datasets = split_dataset_by_court(train_dataset)
batch_size = 1  # Set your desired batch size
train_dataloaders = create_dataloaders(train_court_datasets, batch_size=batch_size, shuffle=True)
train_batches = get_combined_dataloader(train_dataloaders)


print("Splitting validation data by court...")
eval_court_datasets = split_dataset_by_court(eval_dataset)
eval_dataloaders = create_dataloaders(eval_court_datasets, batch_size=batch_size, shuffle=False)
eval_batches = get_combined_dataloader(eval_dataloaders)


# Define training parameters
num_epochs = 5
learning_rate = 5e-5
accumulation_steps = 4  # Adjust as needed

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
scaler = GradScaler()

# Training loop
total_steps = num_epochs * len(train_batches)
step_count = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    random.shuffle(train_batches)
    pbar = tqdm(enumerate(train_batches), total=len(train_batches), desc=f"Epoch {epoch+1}")
    for step, batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        adapter_names_list = batch['adapter_names']

        batch_adapter_names = adapter_names_list[0]
        consistent_adapters = all(adapter_names == batch_adapter_names for adapter_names in adapter_names_list)
        if not consistent_adapters:
            raise ValueError("All samples in the batch must have the same adapter configurations.")

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                name_parts = name.split('.')
                if 'h' in name_parts:
                    h_idx = name_parts.index('h')
                    layer_idx = name_parts[h_idx + 1]
                    adapter_name = batch_adapter_names.get(layer_idx, None)
                    if adapter_name is not None:
                        if adapter_name in module.lora_A:
                            module.set_adapter(adapter_name)
                        else:
                            raise ValueError(f"Adapter '{adapter_name}' not found in module '{name}'.")

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        loss_value = loss.item()  # Get the loss value before dividing

        loss = loss / accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss_value  # Accumulate the original loss value

        step_count += 1

        # Print loss every 2 steps
        if (step + 1) % 2 == 0:
            print(f"Step {step_count}, Loss: {loss_value:.4f}")

        # Update tqdm progress bar
        pbar.set_postfix({'loss': f"{loss_value:.4f}"})

    # Handle remaining gradients
    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    average_loss = epoch_loss / len(train_batches)
    print(f"Epoch {epoch +1}, Average Loss: {average_loss:.4f}")

# Evaluation loop
print("\nStarting evaluation...")
model.eval()
eval_loss = 0.0
eval_steps = 0

with torch.no_grad():
    for batch in tqdm(eval_batches, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        adapter_names_list = batch['adapter_names']

        batch_adapter_names = adapter_names_list[0]
        consistent_adapters = all(adapter_names == batch_adapter_names for adapter_names in adapter_names_list)
        if not consistent_adapters:
            raise ValueError("All samples in the batch must have the same adapter configurations.")

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                name_parts = name.split('.')
                if 'h' in name_parts:
                    h_idx = name_parts.index('h')
                    layer_idx = name_parts[h_idx + 1]
                    adapter_name = batch_adapter_names.get(layer_idx, None)
                    if adapter_name is not None:
                        if adapter_name in module.lora_A:
                            module.set_adapter(adapter_name)
                        else:
                            raise ValueError(f"Adapter '{adapter_name}' not found in module '{name}'.")

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        eval_loss += loss.item()
        eval_steps += 1

average_eval_loss = eval_loss / eval_steps
print(f"Perplexity: {math.exp(average_eval_loss):.4f}")
print(f"Evaluation Loss: {average_eval_loss:.4f}")


# # Compare parameters after training and print modules that didn't change
# print("\nModules that didn't have their values changed during training:")
# unchanged_modules = set()

# for name, param in model.named_parameters():
#     initial_param = initial_params[name]
#     # if param.requires_grad:
#     if torch.equal(param.cpu().data, initial_param.cpu().data):
#         module_name = param_to_module[name]
#         unchanged_modules.add(module_name)

# # Print the list of modules with unchanged parameters
# for module_name in sorted(unchanged_modules):
#     print(module_name)
# Proof of concept