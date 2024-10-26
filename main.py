# Install required libraries if you haven't already
# !pip install transformers datasets peft

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from datasets import load_dataset
import torch
import json

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

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

# print(model)

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
                lora_dropout = module.lora_dropout[adapter_name].p if isinstance(module.lora_dropout[adapter_name], torch.nn.Dropout) else 0.0
                print(f"    * Adapter Name: {adapter_name}")
                print(f"      - Rank (r): {r}")
                print(f"      - Alpha: {lora_alpha}")
                print(f"      - Dropout: {lora_dropout}")
            print()

print_model_adapters(model)

# # Enhanced tokenization function that creates labels for causal LM
# def tokenize_and_label(examples):
#     # Tokenize the texts
#     tokenized_inputs = tokenizer(
#         examples['text'],
#         truncation=True,
#         padding='max_length',
#         max_length=128,
#         return_tensors='pt'
#     )
    
#     # Create labels (for causal LM, labels are the same as input_ids)
#     labels = tokenized_inputs['input_ids'].clone()
    
#     # Mark padding tokens with -100 so they're ignored in the loss
#     labels[labels == tokenizer.pad_token_id] = -100
    
#     return {
#         'input_ids': tokenized_inputs['input_ids'],
#         'attention_mask': tokenized_inputs['attention_mask'],
#         'labels': labels
#     }

# # Load a small portion of a dataset for demonstration purposes
# dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1%]')
# tokenized_dataset = dataset.map(
#     tokenize_and_label,
#     batched=True,
#     remove_columns=dataset.column_names
# )
# tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     logging_steps=10,
#     eval_strategy="epoch",
#     fp16=True,
#     remove_unused_columns=False,
#     # gradient_accumulation_steps=4,    # Added for better memory management
#     learning_rate=2e-4,              # Added specific learning rate
#     warmup_steps=100,                # Added warmup steps
#     weight_decay=0.01,               # Added weight decay
#     save_strategy="no",      # Disable all saving
#     save_steps=None,         # No saving at specific steps
#     save_total_limit=0,      # Don't keep any checkpoints
#     report_to="none",        # Disable wandb/tensorboard/etc logging
# )

# # Initialize the Trainer with the custom data collator
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     eval_dataset=tokenized_dataset
# )

# # Start training
# trainer.train()