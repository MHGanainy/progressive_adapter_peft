import torch
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import multiprocessing
import math
import os

# 0. Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For Transformer models
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Ensure deterministic behavior for CUDA operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

# 1. Initialize the tokenizer and model
model_name = "gpt2-xl"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

# Load pre-trained GPT-2 XL model
# Since you've modified the model classes in the transformers library,
# we can load the model directly
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

# 2. Load your dataset
dataset = load_dataset('MHGanainy/multi_clustering', 'lex-former-8-clustered-instance-b-dataset-cluster')

block_size = 1024

# Task types mapping
dataset_name_to_task_types = {
    'santoshtyss/uk_courts_cases': [0, 1, 3, 3],
    'santoshtyss/eu-court-cases': [0, 0, 1, 1],
    'santoshtyss/indian_courts_cases': [0, 1, 2, 2],
    'santoshtyss/ecthr_cases': [0, 0, 0, 0],
    'santoshtyss/canadian_court_cases': [0, 1, 3, 4]
}

# 3. Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the text
    result = tokenizer(
        examples["original_text"],
        padding='max_length',
        max_length=block_size,
        truncation=True,
    )
    input_ids = result["input_ids"]

    # Copy input_ids to labels
    labels = input_ids.copy()

    # Set the first 512 tokens of labels to -100
    labels = [[-100]*512 + ids[512:] for ids in labels]

    # Set labels to -100 where input_ids == pad_token_id
    labels = [
        [label if input_id != tokenizer.pad_token_id else -100 for input_id, label in zip(input_ids_seq, labels_seq)]
        for input_ids_seq, labels_seq in zip(input_ids, labels)
    ]
    result["labels"] = labels

    # Map the dataset_name to task_types
    dataset_names = examples['dataset_name']
    task_types_list = [
        dataset_name_to_task_types.get(name, [0, 0, 0, 0]) if name in dataset_name_to_task_types else print(f"{name} not found") or [0, 0, 0, 0]
        for name in dataset_names
    ]
    result['task_types'] = task_types_list

    return result

# Apply the tokenize_function to the dataset
def prepare_dataset(dataset_split, split="train"):
    total_cores = multiprocessing.cpu_count()
    num_cpu_cores = min(64, total_cores)
    print(f"Using {num_cpu_cores} CPU cores for '{split}' dataset processing.")

    lm_dataset = dataset_split.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset_split.column_names,
        desc=f"Tokenizing {split} dataset",
        num_proc=num_cpu_cores,
        # Ensure deterministic shuffling
        load_from_cache_file=True,
        writer_batch_size=1000,
    )

    lm_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'task_types'])
    return lm_dataset

print("Preprocessing training data...")
train_dataset = prepare_dataset(dataset["train"], "train")

print("Preprocessing validation data...")
eval_dataset = prepare_dataset(dataset["validation"], "validation")

# 4. Apply PEFT with LoRA configurations
# Define LoRA configurations
peft_config_layers_0_11 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    layers_to_transform=list(range(0, 12)),
    layers_pattern="h",
    num_adapters_per_layer=1,
    layer_group=0,
)

peft_config_layers_12_23 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    layers_to_transform=list(range(12, 24)),
    layers_pattern="h",
    num_adapters_per_layer=2,
    layer_group=1,
)

peft_config_layers_24_35 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    layers_to_transform=list(range(24, 36)),
    layers_pattern="h",
    num_adapters_per_layer=4,
    layer_group=2,
)

peft_config_layers_36_47 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=13,
    lora_alpha=26,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    layers_to_transform=list(range(36, 48)),
    layers_pattern="h",
    num_adapters_per_layer=5,
    layer_group=3,
)

# Apply PEFT to the model
model = get_peft_model(model, peft_config_layers_0_11, adapter_name="layer_0_11")
model.add_adapter("layer_12_23", peft_config_layers_12_23)
model.add_adapter("layer_24_35", peft_config_layers_24_35)
model.add_adapter("layer_36_47", peft_config_layers_36_47)

# Manually set requires_grad=True for all adapter parameters
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

model.print_trainable_parameters()  # Optional: Print trainable parameters

# 5. Define training arguments
batch_size: int = 4
num_train_epochs = 1
steps_per_epoch = len(train_dataset) // batch_size
total_steps = int(steps_per_epoch * num_train_epochs)

training_args = TrainingArguments(
    output_dir=f"./gpt2-xl-peft-lora",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    warmup_steps=int(0.1 * total_steps),
    logging_steps=100,
    fp16=True,  # Use mixed precision training
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    ddp_find_unused_parameters=True,
    save_strategy="no",
    report_to="none",
    seed=seed,  # Set seed for TrainingArguments
    data_seed=seed,  # Ensure data shuffling is reproducible
)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_steps}")

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 7. Start training
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

metrics = trainer.evaluate()
perplexity = math.exp(metrics["eval_loss"])
metrics["perplexity"] = perplexity

trainer.log_metrics("eval", metrics)
