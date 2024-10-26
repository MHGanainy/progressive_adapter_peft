# Install required libraries if you haven't already
# !pip install transformers datasets peft

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Load GPT-2 XL tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

# Set up LoRA configuration with correct target modules
peft_config_1 = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['c_attn', 'c_proj'],
    bias='none',
    task_type='CAUSAL_LM'
)

peft_config_2 = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['c_attn', 'c_proj'],
    bias='none',
    task_type='CAUSAL_LM'
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Enhanced tokenization function that creates labels for causal LM
def tokenize_and_label(examples):
    # Tokenize the texts
    tokenized_inputs = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Create labels (for causal LM, labels are the same as input_ids)
    labels = tokenized_inputs['input_ids'].clone()
    
    # Mark padding tokens with -100 so they're ignored in the loss
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': labels
    }

# Load a small portion of a dataset for demonstration purposes
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1%]')
tokenized_dataset = dataset.map(
    tokenize_and_label,
    batched=True,
    remove_columns=dataset.column_names
)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_steps=10,
    eval_strategy="epoch",
    fp16=True,
    remove_unused_columns=False,
    # gradient_accumulation_steps=4,    # Added for better memory management
    learning_rate=2e-4,              # Added specific learning rate
    warmup_steps=100,                # Added warmup steps
    weight_decay=0.01,               # Added weight decay
    save_strategy="no",      # Disable all saving
    save_steps=None,         # No saving at specific steps
    save_total_limit=0,      # Don't keep any checkpoints
    report_to="none",        # Disable wandb/tensorboard/etc logging
)

# Initialize the Trainer with the custom data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset
)

# Start training
trainer.train()