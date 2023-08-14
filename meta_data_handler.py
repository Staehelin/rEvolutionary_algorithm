import config


def save_meta_data(meta_data_dictionary):
    with open(config.META_DATA_FILENAME, 'w') as f:
        for key, value in meta_data_dictionary.items():
            f.write(f'{key}: {value}\n')


def load_meta_data():
    metadata_filename = config.META_DATA_FILENAME

    # Load metadata if the file exists
    loaded_metadata = {}
    try:
        with open(metadata_filename, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ', 1)
                loaded_metadata[key] = value
        print("Metadata loaded.")
    except FileNotFoundError:
        print("Metadata file not found.")

    return loaded_metadata

