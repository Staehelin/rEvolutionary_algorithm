import config
import json

meta_data = {
    "generation" : 0,
    "generations_list" : [],
    "generations_fitness_list": [],
    "project_title" : config.PROJECT_TITLE,
    "network_architechture": {
        'input_size': config.INPUT_SIZE,
        'hidden_layers': config.HIDDEN_LAYERS,
        'nodes_per_layer': config.NODES_PER_LAYER,
        'output_size': config.OUTPUT_SIZE,
        'activation_hidden': config.ACTIVATION_HIDDEN,
        'activation_output': config.ACTIVATION_OUTPUT,
        'initializer_seed': config.INITIALIZER_SEED,
    },
}

def save_meta_data():
    metadata_filename = config.META_DATA_FILENAME
    global meta_data

    with open(metadata_filename, 'w') as f:
        json.dump(meta_data, f, indent=4)
    print("Metadata saved.")


def load_meta_data():
    metadata_filename = config.META_DATA_FILENAME

    try:
        with open(metadata_filename, 'r') as f:
            loaded_metadata = json.load(f)
        print("Metadata loaded.")
        return loaded_metadata
    except FileNotFoundError:
        print("Metadata file not found.")



