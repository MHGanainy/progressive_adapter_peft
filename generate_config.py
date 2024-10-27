import json

COURT_TREE = {
    "AC": {
        "description": "All Courts",
        "rank": 64,
        "children": {
            "EC": {
                "description": "European Courts",
                "rank": 32,
                "children": {
                    "ECHR": {
                        "description": "European Court of Human Rights",
                        "rank": 16,
                        "children": {}
                    },
                    "EUC": {
                        "description": "EU Courts",
                        "rank": 16,
                        "children": {}
                    }
                }
            },
            "CC": {
                "description": "Commonwealth Courts",
                "rank": 32,
                "children": {
                    "IC": {
                        "description": "Indian Courts",
                        "rank": 16,
                        "children": {}
                    },
                    "ACC": {
                        "description": "Anglo-Canadian Courts",
                        "rank": 16,
                        "children": {
                            "UKC": {
                                "description": "UK Courts",
                                "rank": 13,
                                "children": {}
                            },
                            "CAC": {
                                "description": "Canadian Courts",
                                "rank": 13,
                                "children": {}
                            }
                        }
                    }
                }
            }
        }
    }
}

def generate_lora_config():
    adapter_configs = []
    layer_assignments = {}

    # Group 1: Layers 0-11, AC, rank 64
    group1_layers = list(range(0, 12))
    layer_assignments['AC'] = group1_layers
    for layer in group1_layers:
        adapter_configs.append({
            "adapter_name": f"layer_{layer}_adapter_AC",
            "layers_to_transform": [layer],
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.1
        })

    # Group 2: Layers 12-23, EC and CC, rank 32 each
    group2_layers = list(range(12, 24))
    for court in ['EC', 'CC']:
        layer_assignments[court] = group2_layers
        for layer in group2_layers:
            adapter_configs.append({
                "adapter_name": f"layer_{layer}_adapter_{court}",
                "layers_to_transform": [layer],
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05
            })

    # Group 3: Layers 24-35, ECHR, EUC, IC, ACC, rank 16 each
    group3_layers = list(range(24, 36))
    for court in ['ECHR', 'EUC', 'IC', 'ACC']:
        layer_assignments.setdefault(court, []).extend(group3_layers)
        for layer in group3_layers:
            adapter_configs.append({
                "adapter_name": f"layer_{layer}_adapter_{court}",
                "layers_to_transform": [layer],
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0
            })

    # Group 4: Layers 36-47, ECHR, EUC, IC, UKC, CAC, rank 13 each
    group4_layers = list(range(36, 48))
    for court in ['ECHR', 'EUC', 'IC', 'UKC', 'CAC']:
        layer_assignments.setdefault(court, []).extend(group4_layers)
        for layer in group4_layers:
            adapter_configs.append({
                "adapter_name": f"layer_{layer}_adapter_{court}",
                "layers_to_transform": [layer],
                "r": 13,
                "lora_alpha": 26,
                "lora_dropout": 0.0
            })

    # Save adapter configurations to JSON file
    config = {"adapter_configs": adapter_configs}
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Generated configuration for {len(adapter_configs)} adapters")
    print("Layer assignments per node:")
    for node_name, layers in layer_assignments.items():
        print(f"- {node_name}: Layers {layers}")

    print_layer_assignments(layer_assignments=layer_assignments, adapter_configs=adapter_configs)
    # Return layer assignments
    return layer_assignments

def collect_all_courts(node, path=None, courts=None):
    if path is None:
        path = []
    if courts is None:
        courts = []
    node_name = path[-1] if path else 'AC'
    if node_name not in courts:
        courts.append(node_name)
    if 'children' in node and node['children']:
        for child_name, child_node in node['children'].items():
            collect_all_courts(child_node, path + [child_name], courts)
    return courts

def generate_adapter_mappings():
    # First generate layer assignments
    layer_assignments = generate_lora_config()
    # Collect all courts
    all_courts = collect_all_courts(COURT_TREE['AC'], ['AC'])
    # Generate adapter mappings for each court
    adapter_mappings = {}
    for court in all_courts:
        adapter_mapping = get_adapter_mapping_for_court(court, layer_assignments)
        # Convert layer indices to strings for JSON serialization
        adapter_mapping_str = {str(k): v for k, v in adapter_mapping.items()}
        adapter_mappings[court] = adapter_mapping_str
    # Dump adapter mappings to JSON file
    with open('adapter_mappings.json', 'w') as f:
        json.dump(adapter_mappings, f, indent=4)
    print("Adapter mappings have been saved to 'adapter_mappings.json'.")

def get_adapter_mapping_for_court(target_court, layer_assignments):
    adapter_names = {}

    # Helper function to find the path to the target court
    def find_path(node, path):
        node_name = path[-1]
        if node_name == target_court:
            return path
        if 'children' in node and node['children']:
            for child_name, child_node in node['children'].items():
                result = find_path(child_node, path + [child_name])
                if result:
                    return result
        return None

    # Find the path from root to target court
    path = find_path(COURT_TREE['AC'], ['AC'])
    if not path:
        raise ValueError(f"Target court '{target_court}' not found in the court tree.")

    # Build the adapter names mapping
    for node_name in path:
        layers = layer_assignments.get(node_name, [])
        for layer in layers:
            adapter_names[layer] = f"layer_{layer}_adapter_{node_name}"

    # Include adapters assigned directly to the target court (if any)
    if target_court in layer_assignments and target_court not in path:
        layers = layer_assignments[target_court]
        for layer in layers:
            adapter_names[layer] = f"layer_{layer}_adapter_{target_court}"

    return adapter_names

def get_adapter_mapping(court_name):
    """
    Lookup method to retrieve the adapter mapping for a given court from the JSON file.
    """
    with open('adapter_mappings.json', 'r') as f:
        adapter_mappings = json.load(f)
    if court_name in adapter_mappings:
        # Convert keys back to integers (they are saved as strings in JSON)
        adapter_mapping = {int(k): v for k, v in adapter_mappings[court_name].items()}
        return adapter_mapping
    else:
        raise ValueError(f"Court '{court_name}' not found in adapter mappings.")

def print_layer_assignments(layer_assignments, adapter_configs):
    print(f"\nGenerated configuration for {len(adapter_configs)} adapters")
    
    def get_rank(node, layer):
        for config in adapter_configs:
            if config['adapter_name'] == f"layer_{layer}_{node}":
                return config['r']
        return None

    def get_layer_ranges(node_name, layers):
        ranges = []
        if not layers:
            return ranges
            
        current_rank = get_rank(node_name, layers[0])
        start_layer = layers[0]
        
        for i in range(1, len(layers)):
            new_rank = get_rank(node_name, layers[i])
            if new_rank != current_rank:
                ranges.append((start_layer, layers[i-1], current_rank))
                current_rank = new_rank
                start_layer = layers[i]
        
        ranges.append((start_layer, layers[-1], current_rank))
        return ranges

    # Print tree structure
    print("\nCourt System Tree with Layer Assignments:")
    print("\nLayer Groups:")
    print("└── Group 1: Layers 0-11")
    print("└── Group 2: Layers 12-23")
    print("└── Group 3: Layers 24-35")
    print("└── Group 4: Layers 36-47")
    print("\nTree Structure with Layer Assignments:")
    
    # Root level
    ac_ranges = get_layer_ranges("AC", layer_assignments["AC"])
    print("\nAll Courts (AC)")
    for start, end, rank in ac_ranges:
        print(f"├── Group {(start//12)+1}: Layers {start}-{end} [rank {rank}]")
    
    # European Courts branch
    print("│   │")
    print("├── European Courts (EC)")
    ec_ranges = get_layer_ranges("EC", layer_assignments["EC"])
    for start, end, rank in ec_ranges:
        print(f"│   ├── Group {(start//12)+1}: Layers {start}-{end} [rank {rank}]")
    
    # EC children
    for court in ["ECHR", "EUC"]:
        print(f"│   │   │")
        print(f"│   ├── {court}")
        ranges = get_layer_ranges(court, layer_assignments[court])
        for start, end, rank in ranges:
            print(f"│   │   ├── Group {(start//12)+1}: Layers {start}-{end} [rank {rank}]")
    
    # Commonwealth Courts branch
    print("│")
    print("└── Commonwealth Courts (CC)")
    cc_ranges = get_layer_ranges("CC", layer_assignments["CC"])
    for start, end, rank in cc_ranges:
        print(f"    ├── Group {(start//12)+1}: Layers {start}-{end} [rank {rank}]")
    
    # CC children
    for court in ["IC", "ACC"]:
        print(f"    │   │")
        print(f"    ├── {court}")
        ranges = get_layer_ranges(court, layer_assignments[court])
        for start, end, rank in ranges:
            print(f"    │   ├── Group {(start//12)+1}: Layers {start}-{end} [rank {rank}]")
        
        if court == "ACC":
            # ACC children
            for sub_court in ["UKC", "CAC"]:
                print(f"    │   │   │")
                print(f"    │   ├── {sub_court}")
                ranges = get_layer_ranges(sub_court, layer_assignments[sub_court])
                for start, end, rank in ranges:
                    print(f"    │   │   ├── Group {(start//12)+1}: Layers {start}-{end} [rank {rank}]")

# Example usage
if __name__ == "__main__":
    # Generate and dump the adapter mappings
    generate_adapter_mappings()

    # Example: Get adapter mapping for 'ECHR'
    # court = 'ECHR'
    # adapter_mapping = get_adapter_mapping(court)
    # # Sort by layer for display
    # adapter_mapping = dict(sorted(adapter_mapping.items()))
    # print(f"\nAdapter mapping for '{court}':")
    # for layer, adapter_name in adapter_mapping.items():
    #     print(f"{layer}: '{adapter_name}'")
