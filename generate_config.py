import json

def generate_lora_config():
    adapter_configs = []
    
    # Layers 0-15: 1 LoRA with rank 64
    for layer in range(16):
        adapter_configs.append({
            "adapter_name": f"layer_{layer}_adapter_0",
            "layers_to_transform": [layer],
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.1
        })
    
    # Layers 16-31: 2 LoRAs with rank 32 each
    for layer in range(16, 32):
        for adapter_idx in range(2):
            adapter_configs.append({
                "adapter_name": f"layer_{layer}_adapter_{adapter_idx}",
                "layers_to_transform": [layer],
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.1
            })
    
    # Layers 32-47: 2 LoRAs with rank 16 each
    for layer in range(32, 48):
        for adapter_idx in range(4):
            adapter_configs.append({
                "adapter_name": f"layer_{layer}_adapter_{adapter_idx}",
                "layers_to_transform": [layer],
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1
            })
    
    config = {"adapter_configs": adapter_configs}
    
    # Save to JSON file with proper formatting
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Generated configuration for {len(adapter_configs)} adapters")
    print("Configuration saved to config.json")

if __name__ == "__main__":
    generate_lora_config()