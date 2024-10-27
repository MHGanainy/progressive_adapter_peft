from tqdm import tqdm
import torch
import random
import numpy as np
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

# Define 5 samples with adapter_names
samples_data = [
    {
        'text': 'This is a case from the European Court of Human Rights regarding freedom of speech.',
        'court': 'ECHR'
    },
    {
        'text': 'The Canadian Supreme Court ruled on the new environmental regulations.',
        'court': 'CAC'
    },
    {
        'text': 'Recent judgments from the UK Supreme Court have significant implications.',
        'court': 'UKC'
    },
    {
        'text': 'The Indian High Courts have issued new guidelines on data privacy.',
        'court': 'IC'
    },
    {
        'text': 'The EU Courts have announced a landmark decision affecting trade laws.',
        'court': 'EUC'
    },
]

# Create a dataset using the datasets library
dataset = Dataset.from_list(samples_data)

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

    # Generate adapter names based on the court field
    adapter_names = []
    for court in examples['court']:
        adapter_mapping = get_adapter_mapping(court)
        # Convert layer indices to strings for consistency
        adapter_names.append({str(k): v for k, v in adapter_mapping.items()})

    result = {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': labels,
        'adapter_names': adapter_names,
        # Exclude 'text' and 'court' from the result
    }
    return result

# Tokenize the dataset and remove the 'text' and 'court' fields
dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'court'])

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
# Set batch_size=1 to handle different adapter configurations per sample
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

# Define training parameters
num_epochs = 5
learning_rate = 5e-5

# Filter the model parameters to include only those of the adapters
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize the GradScaler
scaler = GradScaler()

# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        adapter_names_list = batch['adapter_names']  # List of adapter_names dicts

        # Since batch_size=1, take the adapter_names from the single sample
        batch_adapter_names = adapter_names_list[0]  # Dict mapping layer_idx to adapter_name

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
                        # You can choose to set a default adapter or skip
                        continue

        with autocast():
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

        # Since batch_size=1, take the adapter_names from the single sample
        batch_adapter_names = adapter_names_list[0]  # Dict mapping layer_idx to adapter_name

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
                        # You can choose to set a default adapter or skip
                        continue

        with autocast():
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
print("Perplexity: ", math.exp(average_eval_loss))
print(f"Evaluation Loss: {average_eval_loss}")


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