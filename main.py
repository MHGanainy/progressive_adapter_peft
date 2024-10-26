# Install required libraries if you haven't already
# !pip install transformers datasets peft

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import json

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

# Print the adapters in the model
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

# Define the single sample with adapter_names
sample = {
    'text': 'Hello, how are you?',
}

# Create a custom dataset with one sample
class SingleSampleDataset(Dataset):
    def __init__(self, sample, tokenizer):
        self.sample = sample
        self.tokenizer = tokenizer

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        text = self.sample['text']
        adapter_names = torch.tensor([9000 + (layer_idx * 10) for layer_idx in range(12)])

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
dataset = SingleSampleDataset(sample, tokenizer)

# Create a wrapper for the model
class CustomModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def decode_adapter_names(self, adapter_names_tensor):
        """
        Converts tensor of shape [batch_size, num_layers] to list of dictionaries
        Each value is converted from format "9XXY" to "layer_XX_adapter_Y"
        """
        # Move tensor to CPU and convert to string format
        batch_size, num_layers = adapter_names_tensor.shape
        adapter_names_list = []
        
        # Process each item in the batch
        for batch_idx in range(batch_size):
            adapter_dict = {}
            # Process each layer
            for layer_idx in range(num_layers):
                code = str(adapter_names_tensor[batch_idx, layer_idx].item())  # e.g., "9010"
                layer_num = code[1:3]  # "01"
                adapter_num = code[3]   # "0"
                adapter_dict[int(layer_num)] = f"layer_{int(layer_num)}_adapter_{int(adapter_num)}"
            
            adapter_names_list.append(adapter_dict)
            
        # If batch size is 1, return just the dictionary instead of a list
        return adapter_names_list

    def forward(self, input_ids=None, attention_mask=None, labels=None, adapter_names=None, **kwargs):
        # Remove adapter_names from device if it exists
        if adapter_names is not None:
            # Convert the tensor format to dictionary format
            adapter_dict = self.decode_adapter_names(adapter_names)
            # Remove the tensor format from inputs
            adapter_names = None
            # Add the dictionary format to kwargs
            kwargs['adapter_names'] = adapter_dict

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

model = CustomModelWrapper(model)



# Subclass Trainer to pass adapter_names
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract adapter_names from inputs
        # Ensure adapter_names is included when calling model
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, tuple) else outputs['loss']
        return (loss, outputs) if return_outputs else loss

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_steps=10,
    save_total_limit=2,
    learning_rate=5e-5,
    remove_unused_columns=False,  # Important to prevent dropping 'adapter_names',
)

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()
