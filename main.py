# Install required libraries if you haven't already
# !pip install transformers datasets peft
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
import json
import math
import warnings
from typing import Any, Optional, Union
from torch import nn
import copy  # Import copy module for deep copying

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2-xl')

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

# Save initial parameters before training
initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

# Create a mapping from parameter names to module names
param_to_module = {}
for module_name, module in model.named_modules():
    for param_name, param in module.named_parameters(recurse=False):
        full_param_name = f"{module_name}.{param_name}" if module_name else param_name
        param_to_module[full_param_name] = module_name

# Define 4 dummy samples with adapter_names
samples = [
    # First batch samples (Batch 1)
    {
        'text': 'Hello, how are you?',
        'adapter_names': {
            # Layers 0-3 use 'layer_0_adapter_0' in all samples
            **{str(i): f'layer_{i}_adapter_0' for i in range(16)},  # layers 0-3
            **{str(i): f'layer_{i}_adapter_0' for i in range(16, 48)}
        }
    },
    {
        'text': 'What is the weather today?',
        'adapter_names': {
            **{str(i): f'layer_{i}_adapter_0' for i in range(16)},  # layers 0-3
            **{str(i): f'layer_{i}_adapter_0' for i in range(16, 48)}
        }
    },
    # Second batch samples (Batch 2)
    {
        'text': 'Tell me a joke.',
        'adapter_names': {
            **{str(i): f'layer_{i}_adapter_0' for i in range(16)},  # layers 0-3
            **{str(i): f'layer_{i}_adapter_1' for i in range(16, 48)}
        }
    },
    {
        'text': 'How does a computer work?',
        'adapter_names': {
            **{str(i): f'layer_{i}_adapter_0' for i in range(16)},  # layers 0-3
            **{str(i): f'layer_{i}_adapter_1' for i in range(16, 48)}
        }
    },
]

# Create a dataset using the datasets library
dataset = Dataset.from_list(samples)

# Define the tokenize function
def tokenize_function(examples):
    # Tokenize the text
    encoding = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128,
    )
    # Labels are the same as input_ids for language modeling
    labels = encoding['input_ids']

    result = {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': labels,
        'adapter_names': examples['adapter_names'],
        # Exclude 'text' from the result
    }
    return result

# Tokenize the dataset and remove the 'text' field
dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Set the format of the dataset
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)

# Define the data collator
class DataCollatorWithAdapterNames(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Extract adapter_names and remove from features
        adapter_names = [feature.pop('adapter_names') for feature in features]

        # Use the parent class method to collate input_ids and labels
        batch = super().__call__(features)

        # Add adapter_names back to the batch
        batch['adapter_names'] = adapter_names

        return batch

data_collator = DataCollatorWithAdapterNames(tokenizer=tokenizer, mlm=False)

# Create the DataLoader
# Set batch_size=2 to batch samples with the same adapter configurations
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=data_collator)

# Define training parameters
num_epochs = 5
learning_rate = 5e-5

# Filter the model parameters to include only those of the adapters
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        adapter_names_list = batch['adapter_names']  # List of adapter_names dicts

        # Since all samples in the batch have the same adapter configurations,
        # we can take the adapter_names from the first sample
        batch_adapter_names = adapter_names_list[0]  # Dict mapping layer_idx to adapter_name

        # Verify that all samples in the batch have the same adapter_names
        consistent_adapters = all(adapter_names == batch_adapter_names for adapter_names in adapter_names_list)
        if not consistent_adapters:
            raise ValueError("All samples in the batch must have the same adapter configurations.")

        # Set active adapters per layer using set_adapter
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                # Extract layer index from module name
                name_parts = name.split('.')
                if 'h' in name_parts:
                    h_idx = name_parts.index('h')
                    layer_idx = name_parts[h_idx + 1]  # Layer index as string

                    # Get adapter name for this layer
                    adapter_name = batch_adapter_names.get(layer_idx, None)

                    if adapter_name is not None:
                        # Check if adapter_name exists in module.lora_A
                        if adapter_name in module.lora_A:
                            # Set the active adapter using set_adapter method
                            module.set_adapter(adapter_name)
                        else:
                            # The adapter_name does not exist in this module
                            print(f"Adapter '{adapter_name}' not found in module '{name}'. Available adapters: {list(module.lora_A.keys())}")
                            # Raise an error
                            raise ValueError(f"Adapter '{adapter_name}' not found in module '{name}'.")
                    else:
                        # No adapter specified for this layer
                        print(f"No adapter specified for layer {layer_idx} in sample.")
                        # You can choose to set a default adapter or skip
                        continue

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
    if (epoch + 1) % 1 == 0:
        average_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {average_loss}")

# Evaluation loop
print("\nStarting evaluation...")
model.eval()
eval_loss = 0.0
eval_steps = 0

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        adapter_names_list = batch['adapter_names']  # List of adapter_names dicts

        # Since all samples in the batch have the same adapter configurations,
        # we can take the adapter_names from the first sample
        batch_adapter_names = adapter_names_list[0]  # Dict mapping layer_idx to adapter_name

        # Verify that all samples in the batch have the same adapter_names
        consistent_adapters = all(adapter_names == batch_adapter_names for adapter_names in adapter_names_list)
        if not consistent_adapters:
            raise ValueError("All samples in the batch must have the same adapter configurations.")

        # Set active adapters per layer using set_adapter
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                # Extract layer index from module name
                name_parts = name.split('.')
                if 'h' in name_parts:
                    h_idx = name_parts.index('h')
                    layer_idx = name_parts[h_idx + 1]  # Layer index as string

                    # Get adapter name for this layer
                    adapter_name = batch_adapter_names.get(layer_idx, None)

                    if adapter_name is not None:
                        # Check if adapter_name exists in module.lora_A
                        if adapter_name in module.lora_A:
                            # Set the active adapter using set_adapter method
                            module.set_adapter(adapter_name)
                        else:
                            # The adapter_name does not exist in this module
                            print(f"Adapter '{adapter_name}' not found in module '{name}'. Available adapters: {list(module.lora_A.keys())}")
                            # Raise an error
                            raise ValueError(f"Adapter '{adapter_name}' not found in module '{name}'.")
                    else:
                        # No adapter specified for this layer
                        print(f"No adapter specified for layer {layer_idx} in sample.")
                        # You can choose to set a default adapter or skip
                        continue

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        eval_loss += loss.item()
        eval_steps += 1

average_eval_loss = eval_loss / eval_steps
print(f"Evaluation Loss: {average_eval_loss}")

# Compare parameters after training and print modules that didn't change
print("\nModules that didn't have their values changed during training:")
unchanged_modules = set()

for name, param in model.named_parameters():
    initial_param = initial_params[name]
    # if param.requires_grad:
    if torch.equal(param.cpu().data, initial_param.cpu().data):
        module_name = param_to_module[name]
        unchanged_modules.add(module_name)

# Print the list of modules with unchanged parameters
for module_name in sorted(unchanged_modules):
    print(module_name)
# Proof of concept