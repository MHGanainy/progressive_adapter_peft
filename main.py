# Install required libraries if you haven't already
# !pip install transformers datasets peft

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
import json
import math
import warnings
from typing import Any, Optional, Union
from torch import nn

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

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

# Print the adapters in the model (optional)
def print_model_adapters(model):
    """
    Prints the adapters present in each layer of the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            print(f"Layer: {name}")
            print(f"  - Base Layer Type: {type(module.base_layer)}")
            print(f"  - Adapters:")
            for adapter_name in module.lora_A.keys():
                r = module.r[adapter_name]
                lora_alpha = module.lora_alpha[adapter_name]
                lora_dropout = (
                    module.lora_dropout[adapter_name].p
                    if isinstance(module.lora_dropout[adapter_name], torch.nn.Dropout)
                    else 0.0
                )
                print(f"    * Adapter Name: {adapter_name}")
                print(f"      - Rank (r): {r}")
                print(f"      - Alpha: {lora_alpha}")
                print(f"      - Dropout: {lora_dropout}")
            print()

print_model_adapters(model)

def print_active_adapters(model):
    """
    Prints the active adapters for each LoraLayer in the model.
    """
    print("Active adapters in the model:")
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            # Extract layer index from module name (optional, for better readability)
            name_parts = name.split('.')
            layer_info = ""
            if 'h' in name_parts:
                h_idx = name_parts.index('h')
                layer_idx = int(name_parts[h_idx + 1])
                layer_info = f" (Layer {layer_idx})"
            active_adapters = module.active_adapters
            if active_adapters:
                print(f"{name}{layer_info}: {active_adapters}")
            else:
                print(f"{name}{layer_info}: No active adapters")

# Call the method before the training loop
print("Before training:")
print_active_adapters(model)

# Define multiple samples with adapter_names
samples = [
    {
        'text': 'Hello, how are you?',
        'adapter_names': {
            layer_idx: f'layer_{layer_idx}_adapter_0' for layer_idx in range(12)
        }
    },
    {
        'text': 'What is the weather today?',
        'adapter_names': {
            layer_idx: f'layer_{layer_idx}_adapter_0' for layer_idx in range(12)
        }
    },
    # Add more samples as needed
]

# Create a custom dataset with multiple samples
class CustomDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        adapter_names = sample['adapter_names']

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Labels are the same as input_ids for language modeling
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'adapter_names': adapter_names
        }

# Instantiate the dataset
dataset = CustomDataset(samples, tokenizer)

# Define the data collator
class DataCollatorWithAdapterNames(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        # Extract adapter_names and remove from examples
        adapter_names = [example.pop('adapter_names') for example in examples]

        # Use the parent class method to collate input_ids and labels
        batch = super().__call__(examples)

        # Add adapter_names back to the batch
        batch['adapter_names'] = adapter_names

        return batch

data_collator = DataCollatorWithAdapterNames(tokenizer=tokenizer, mlm=False)

# Create the DataLoader
from torch.utils.data import DataLoader

# Set batch_size=1 to handle per-sample adapter activation
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

# Define training parameters
num_epochs = 1
learning_rate = 5e-5

# Filter the model parameters to include only those of the adapters
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        adapter_names_list = batch['adapter_names']

        # Since batch_size=1, we take the first item's adapter_names
        batch_adapter_names = adapter_names_list[0]  # Dict mapping layer_idx to adapter_name

        # Set active adapters per layer using set_adapter
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                # Extract layer index from module name
                name_parts = name.split('.')
                if 'h' in name_parts:
                    h_idx = name_parts.index('h')
                    layer_idx = int(name_parts[h_idx + 1])

                    # Get adapter name for this layer
                    adapter_name = batch_adapter_names[layer_idx]

                    # Set the active adapter using set_adapter method
                    module.set_adapter(adapter_name)
        print(f"\nActive adapters before forward pass (Epoch {epoch+1}):")
        print_active_adapters(model)
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {average_loss}")
