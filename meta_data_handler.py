import config
import json


def save_meta_data(metadata):
    metadata_filename = config.META_DATA_FILENAME

    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
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



