import config
import meta_data_handler
import os

meta_data = {
    "generation" : 0,
    "generations_list" : [],
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


def initialize():
    global meta_data
    if (os.path.isfile(config.META_DATA_FILENAME)):
        meta_data = meta_data_handler.load_meta_data()
    else:
        meta_data_handler.save_meta_data(meta_data)
