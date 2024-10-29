import argparse
import os
import random
import json
import math
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from generate_config import get_adapter_mapping  # Ensure this module is available
from tqdm import tqdm
import traceback
import multiprocessing

# Import bitsandbytes for 8-bit optimizers
import bitsandbytes as bnb
from bitsandbytes.optim import AdamW8bit

# ---------------------- Custom Batch Sampler ----------------------
class DistributedAdapterBatchSampler(Sampler):
    # [Your existing DistributedAdapterBatchSampler code here]
    def __init__(self, adapter_to_indices, batch_size, num_replicas=None, rank=None, shuffle=True, drop_last=False):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.adapter_to_indices = adapter_to_indices  # Save for reshuffling
        self.seed = 42
        self.epoch = 0
        self.create_batches()

    def create_batches(self):
        self.batches = []
        for indices in self.adapter_to_indices.values():
            # Shuffle indices within each adapter group
            if self.shuffle:
                random.shuffle(indices)
            # Create batches
            group_batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
            # Handle incomplete batches
            if self.drop_last and len(group_batches[-1]) < self.batch_size:
                group_batches = group_batches[:-1]
            self.batches.extend(group_batches)

        if self.shuffle:
            random.shuffle(self.batches)

        # Ensure total number of batches is divisible by num_replicas
        total_batches = len(self.batches)
        remainder = total_batches % self.num_replicas
        if remainder != 0:
            # Calculate number of batches to drop
            batches_to_drop = remainder
            # Drop batches from the end
            if self.drop_last:
                if self.rank == 0:
                    print(f"Dropping {batches_to_drop} batches to make total divisible by {self.num_replicas}")
                self.batches = self.batches[:-batches_to_drop]
            else:
                # If not dropping, pad with existing batches to make it divisible
                extra_batches = self.batches[:self.num_replicas - remainder]
                if self.rank == 0:
                    print(f"Adding {len(extra_batches)} batches to make total divisible by {self.num_replicas}")
                self.batches.extend(extra_batches)

        # Partition batches among processes
        self.total_size = len(self.batches)
        self.batches = self.batches[self.rank::self.num_replicas]

    def set_epoch(self, epoch):
        # Set the random seed based on the epoch
        self.epoch = epoch
        random.seed(self.seed + epoch)
        # Recreate batches with new shuffling
        self.create_batches()

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

# ---------------------- Argument Parsing and Setup ----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # Add any required arguments here
    args = parser.parse_args()
    return args

args = parse_args()

# Get local_rank from environment variable
local_rank = int(os.environ.get('LOCAL_RANK', 0))

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
    else:
        rank = 0
        world_size = 1
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    return rank, device

rank, device = setup_distributed()
is_main_process = rank == 0

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------- Model and Tokenizer Setup ----------------------
# Configure quantization (if needed)
# Uncomment the following if you want to use bitsandbytes quantization
"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
"""

# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-xl')
model = AutoModelForCausalLM.from_pretrained(
    'openai-community/gpt2-xl',
    # quantization_config=bnb_config,  # Uncomment if using quantization
    device_map={"": device},  # Map the entire model to the local device
    attn_implementation="flash_attention_2"
)

# Prepare model for k-bit training if using bitsandbytes quantization
# Uncomment the following if you want to use bitsandbytes quantization
"""
model = prepare_model_for_kbit_training(model)
"""

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

# Load adapter configurations from 'config.json'
with open('config.json', 'r') as f:
    loaded_configs = json.load(f)

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

# Initialize the PEFT model and add adapters
peft_config = create_lora_config(adapter_configs[0])
model = get_peft_model(model, peft_config, adapter_name=adapter_configs[0]['adapter_name'])

for adapter_cfg in adapter_configs[1:]:
    peft_config = create_lora_config(adapter_cfg)
    model.add_adapter(adapter_cfg['adapter_name'], peft_config)

if is_main_process:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

# Remove the code that sets requires_grad=True for all adapter parameters
# Allow the model to manage requires_grad based on active adapters
# for name, param in model.named_parameters():
#     if 'lora_' in name:
#         param.requires_grad = True

# ---------------------- Dataset Preparation ----------------------
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
        truncation=True,
        return_tensors='pt'  # Return PyTorch tensors directly
    )
    input_ids = result["input_ids"]
    # Copy input_ids to labels
    labels = input_ids.clone()
    # Set the first 512 tokens of labels to -100
    labels[:, :512] = -100
    # Set labels to -100 where input_ids == pad_token_id
    labels[input_ids == tokenizer.pad_token_id] = -100
    result["labels"] = labels

    # Map dataset_name to court code
    courts = [dataset_name_to_court.get(name, 'Unknown') for name in examples['dataset_name']]
    if 'Unknown' in courts:
        raise ValueError("Found unknown dataset_name in examples.")

    result['court'] = courts  # Keep court code in the result

    # Generate adapter names based on the court field
    adapter_names = []
    for court in courts:
        adapter_mapping = get_adapter_mapping(court)
        if not adapter_mapping:
            raise ValueError(f"Adapter mapping not found for court: {court}")
        # Convert layer indices to strings for consistency
        adapter_names.append({str(k): v for k, v in adapter_mapping.items()})
    result['adapter_names'] = adapter_names

    return result

# ---------------------- Caching Functions ----------------------
def save_dataset(dataset, path):
    torch.save(dataset, path)

def load_cached_dataset(path):
    return torch.load(path)

# Define cache paths
cache_dir = 'cache_datasets'
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(cache_dir, 'train_dataset.pt')
eval_cache_path = os.path.join(cache_dir, 'eval_dataset.pt')

def prepare_dataset(dataset_split, split="train"):
    total_cores = multiprocessing.cpu_count()
    num_cpu_cores = min(64, total_cores)
    if is_main_process:
        print(f"Using {num_cpu_cores} CPU cores for '{split}' dataset processing.")

    cache_path = train_cache_path if split == "train" else eval_cache_path

    if os.path.exists(cache_path):
        if is_main_process:
            print(f"Loading cached {split} dataset from '{cache_path}'...")
        lm_dataset = load_cached_dataset(cache_path)
    else:
        if is_main_process:
            print(f"Processing and caching {split} dataset...")
        lm_dataset = dataset_split.map(
            tokenize_function,
            batched=True,
            remove_columns=['original_text', 'cluster_id', 'dataset_name'],
            desc=f"Tokenizing {split} dataset",
            num_proc=num_cpu_cores,
        )
        lm_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "adapter_names"])
        # Save the processed dataset only on the main process
        if is_main_process:
            save_dataset(lm_dataset, cache_path)
            print(f"Saved cached {split} dataset to '{cache_path}'.")
    return lm_dataset

if is_main_process:
    print("Preprocessing training data...")
train_dataset = prepare_dataset(dataset["train"], "train")

if is_main_process:
    print("Preprocessing validation data...")
eval_dataset = prepare_dataset(dataset["validation"], "validation")

# ---------------------- Data Collator ----------------------
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

# ---------------------- Grouping Indices by Adapter ----------------------
def group_indices_by_adapter(dataset, cache_path):
    # [Your existing group_indices_by_adapter code here]

    # Check if grouped indices are cached
    grouped_cache_path = cache_path.replace('.pt', '_grouped.pt')
    if os.path.exists(grouped_cache_path):
        if is_main_process:
            print(f"Loading cached grouped indices from '{grouped_cache_path}'...")
        adapter_to_indices = load_cached_dataset(grouped_cache_path)
    else:
        adapter_to_indices = {}

        # Attempt to determine the total number of examples for the progress bar
        try:
            total = len(dataset)
        except TypeError:
            total = None  # Dataset does not support len()

        # Initialize tqdm progress bar
        if is_main_process:
            progress_bar = tqdm(enumerate(dataset), total=total, desc='Grouping Adapters', unit='example')
        else:
            progress_bar = enumerate(dataset)

        for idx, example in progress_bar:
            # Serialize adapter names to create a unique key
            adapter_key = json.dumps(example['adapter_names'], sort_keys=True)

            # Initialize the list for this adapter_key if it doesn't exist
            if adapter_key not in adapter_to_indices:
                adapter_to_indices[adapter_key] = []

            # Append the current index to the list
            adapter_to_indices[adapter_key].append(idx)

        # Save the grouped indices only on the main process
        if is_main_process:
            save_dataset(adapter_to_indices, grouped_cache_path)
            print(f"Saved cached grouped indices to '{grouped_cache_path}'.")

    return adapter_to_indices

# ---------------------- DataLoader Creation ----------------------
def create_combined_dataloader(dataset, batch_size, split=None, shuffle=True, drop_last=True):
    # Determine cache path based on dataset split
    cache_path = train_cache_path if split == "train" else eval_cache_path

    adapter_to_indices = group_indices_by_adapter(dataset, cache_path)
    batch_sampler = DistributedAdapterBatchSampler(
        adapter_to_indices,
        batch_size,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=shuffle,
        drop_last=drop_last
    )

    total_cpu_cores = multiprocessing.cpu_count()
    num_replicas = torch.distributed.get_world_size()
    num_workers = total_cpu_cores // num_replicas
    num_workers = max(1, num_workers)

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=0,  # Set to 0 to simplify debugging
    )
    return dataloader

batch_size = 1  # Adjust as needed
accumulation_steps = 1
num_epochs = 1

train_dataloader = create_combined_dataloader(train_dataset, batch_size=batch_size, split="train", shuffle=True, drop_last=True)
eval_dataloader = create_combined_dataloader(eval_dataset, batch_size=batch_size, split="eval", shuffle=False, drop_last=True)

# ---------------------- Optimizer and Scheduler ----------------------
learning_rate = 2e-5

# Use bitsandbytes optimizer for 8-bit AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Wrap the model with DDP
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=True
)

# Update lora_layers after wrapping model in DDP
def get_lora_layers_ddp(model):
    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            name_parts = name.split('.')
            if 'h' in name_parts:
                h_idx = name_parts.index('h')
                layer_idx = name_parts[h_idx + 1]
                lora_layers[layer_idx] = module
    return lora_layers

def get_all_lora_layers(model):
    lora_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            lora_layers.append((name, module))
    return lora_layers


# lora_layers = get_lora_layers_ddp(model.module)
scaler = GradScaler()

# Calculate total optimizer steps
steps_per_epoch = len(train_dataloader)
total_optimizer_steps = steps_per_epoch * num_epochs // accumulation_steps

# Scheduler: Warmup 10% of total steps and use cosine schedule
num_warmup_steps = int(0.1 * total_optimizer_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_optimizer_steps
)

def get_relevant_param_names(model, active_adapter_names):
    relevant_param_names = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            # Extract adapter name from parameter name
            name_parts = name.split('.')
            adapter_name = name_parts[-2]  # Assuming adapter name is before 'weight' or 'bias'
            if adapter_name in active_adapter_names:
                relevant_param_names.append(name)
    return relevant_param_names


# ---------------------- Training Function ----------------------
def distributed_training(model, train_dataloader, optimizer, scheduler, scaler, device, is_main_process, num_epochs, accumulation_steps):
    optimizer_step_count = 0

    if is_main_process:
        pbar = tqdm(total=total_optimizer_steps, desc="Training")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        train_dataloader.batch_sampler.set_epoch(epoch)  # Reshuffle for the new epoch

        for step, batch in enumerate(train_dataloader):
            try:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                adapter_names_list = batch['adapter_names']

                batch_adapter_names = adapter_names_list[0]
                # Ensure all samples have the same adapter_names
                if not all(adapter == batch_adapter_names for adapter in adapter_names_list):
                    raise ValueError("All samples in the batch must have the same adapter configurations.")

                # Get all LoraLayer instances
                lora_layers = get_all_lora_layers(model)

                # Set adapters using precomputed mapping and collect active adapters
                active_adapters_info = []
                for module_name, module in lora_layers:
                    # Extract layer index from module_name
                    name_parts = module_name.split('.')
                    if 'h' in name_parts:
                        h_idx = name_parts.index('h')
                        layer_idx = str(name_parts[h_idx + 1])
                    else:
                        module.enable_adapters(False)
                        continue

                    # Get the active adapter name for this layer
                    adapter_name = batch_adapter_names.get(layer_idx, None)
                    if adapter_name:
                        module.set_adapter(adapter_name)
                        module.enable_adapters(True)
                    else:
                        module.enable_adapters(False)

                    # Collect active adapter info
                    active_adapters_info.append(f"Layer {layer_idx}: {module.active_adapter}")
                
                active_adapter_names = set(batch_adapter_names.values())
                # Get the names of parameters that should have gradients
                relevant_param_names = get_relevant_param_names(model, active_adapter_names)
                for name, param in model.named_parameters():
                    if name in relevant_param_names:
                        # Parameters that should have requires_grad=True
                        assert param.requires_grad, (
                            f"Parameter {name} should have requires_grad=True, but has requires_grad={param.requires_grad}."
                        )
                    else:
                        # Parameters that should have requires_grad=False
                        assert not param.requires_grad, (
                            f"Parameter {name} should have requires_grad=False, but has requires_grad={param.requires_grad}."
                        )
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                loss_value = loss.item()
                epoch_loss += loss_value

                loss = loss / accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    scaler.unscale_(optimizer)
                    
                    active_adapter_names = set(batch_adapter_names.values())
                    # Get the names of parameters that should have gradients
                    relevant_param_names = get_relevant_param_names(model, active_adapter_names)

                    # Verify gradients
                    error_messages = []

                    for name, param in model.named_parameters():
                        if name in relevant_param_names:
                            # Parameters that should have non-zero gradients
                            if param.grad is None or param.grad.abs().sum().item() == 0:
                                error_messages.append(f"Parameter {name} should have non-zero gradient, but has zero gradient.")
                        else:
                            # Parameters that should not have gradients
                            if param.grad is not None and param.grad.abs().sum().item() != 0:
                                error_messages.append(f"Parameter {name} should not have gradient, but has non-zero gradient.")

                    if error_messages:
                        print("Gradient check failed:")
                        for msg in error_messages:
                            print(msg)
                    # Proceed with optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    optimizer_step_count += 1

                    if is_main_process:
                        current_lr = scheduler.get_last_lr()[0]
                        pbar.update(1)
                        pbar.set_postfix({'loss': f"{loss_value:.4f}", 'lr': f"{current_lr:.6f}"})
            except Exception as e:
                # Log the exception in all ranks
                rank = torch.distributed.get_rank()
                print(f"Rank {rank}, Step {step} encountered an error: {e}")
                traceback.print_exc()
                # Ensure all processes are aware of the exception
                torch.distributed.barrier()
                # Clean up
                torch.distributed.destroy_process_group()
                sys.exit(1)

        average_epoch_loss = epoch_loss / len(train_dataloader)
        if is_main_process:
            print(f"Epoch {epoch + 1}, Average Loss: {average_epoch_loss:.4f}")

    if is_main_process:
        pbar.close()

# ---------------------- Evaluation Function ----------------------
def distributed_evaluation(model, eval_dataloader, device, is_main_process):
    """
    Handles the distributed evaluation loop.
    """
    model.eval()

    # Set requires_grad=False for all parameters
    for param in model.parameters():
        param.requires_grad = False

    eval_loss = 0.0
    eval_steps = 0

    # Initialize progress bar only on main process
    if is_main_process:
        print("\nStarting evaluation...")
        eval_progress_bar = tqdm(total=len(eval_dataloader), desc="Evaluating")
    else:
        eval_progress_bar = None

    with torch.no_grad():
        for batch in eval_dataloader:
            try:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                adapter_names_list = batch['adapter_names']

                batch_adapter_names = adapter_names_list[0]
                if not all(adapter == batch_adapter_names for adapter in adapter_names_list):
                    raise ValueError("All samples in the batch must have the same adapter configurations.")

                # Get all LoraLayer instances
                lora_layers = get_all_lora_layers(model)

                # Set adapters using precomputed mapping and collect active adapters
                active_adapters_info = []
                for module_name, module in lora_layers:
                    # Extract layer index from module_name
                    name_parts = module_name.split('.')
                    if 'h' in name_parts:
                        h_idx = name_parts.index('h')
                        layer_idx = str(name_parts[h_idx + 1])
                    else:
                        module.enable_adapters(False)
                        continue

                    # Get the active adapter name for this layer
                    adapter_name = batch_adapter_names.get(layer_idx, None)
                    if adapter_name:
                        module.set_adapter(adapter_name)
                        module.enable_adapters(False)

                    # Collect active adapter info
                    active_adapters_info.append(f"Layer {layer_idx}: {module.active_adapter}")

                # Optional: Print active adapters per layer
                # print("Active adapters per layer:")
                # print(", ".join(active_adapters_info))

                # Verify that all parameters have requires_grad=False
                requires_grad_errors = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        requires_grad_errors.append(
                            f"Parameter {name} should have requires_grad=False during evaluation, but has requires_grad=True."
                        )

                if requires_grad_errors:
                    print("requires_grad check failed during evaluation:")
                    for msg in requires_grad_errors:
                        print(msg)

                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                loss_value = loss.item()
                eval_loss += loss_value
                eval_steps += 1

                if is_main_process and eval_progress_bar:
                    eval_progress_bar.update(1)

            except Exception as e:
                # Log the exception in all ranks
                rank = torch.distributed.get_rank()
                print(f"Rank {rank} encountered an error during evaluation: {e}")
                traceback.print_exc()
                # Ensure all processes are aware of the exception
                torch.distributed.barrier()
                # Clean up
                torch.distributed.destroy_process_group()
                sys.exit(1)

    # Create tensors for distributed reduction
    eval_loss_tensor = torch.tensor(eval_loss, device=device)
    eval_steps_tensor = torch.tensor(eval_steps, device=device)

    # Gather results from all processes
    torch.distributed.all_reduce(eval_loss_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(eval_steps_tensor, op=torch.distributed.ReduceOp.SUM)

    # Ensure all processes are synced before calculating final metrics
    torch.distributed.barrier()

    # Compute average_eval_loss (now available to all processes)
    average_eval_loss = eval_loss_tensor.item() / eval_steps_tensor.item()

    if is_main_process:
        if eval_progress_bar:
            eval_progress_bar.close()

        perplexity = math.exp(average_eval_loss)

        print(f"Perplexity: {perplexity:.4f}")
        print(f"Evaluation Loss: {average_eval_loss:.4f}")

    # Final barrier to ensure all processes are done
    torch.distributed.barrier()

    return average_eval_loss

# ---------------------- Main Execution ----------------------
def main():
    try:
        # Training and Evaluation
        distributed_training(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            is_main_process=is_main_process,
            num_epochs=num_epochs,
            accumulation_steps=accumulation_steps
        )

        average_eval_loss = distributed_evaluation(
            model=model,
            eval_dataloader=eval_dataloader,
            device=device,
            is_main_process=is_main_process
        )

    except Exception as e:
        # Exception handling
        rank = torch.distributed.get_rank()
        print(f"Rank {rank} encountered an exception: {e}")
        traceback.print_exc()
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        sys.exit(1)

    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
