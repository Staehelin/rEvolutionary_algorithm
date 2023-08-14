import config
import meta_data_handler
import os
import selector



def initialize():
    if (os.path.isfile(config.META_DATA_FILENAME)):
        meta_data = meta_data_handler.load_meta_data()
    else:
        meta_data_handler.save_meta_data(meta_data_handler.meta_data)

def get_starting_generation():
    if(meta_data_handler.meta_data['generations_list'] == []):
        return selector.get_first_generation()
    else:
        return selector.load_latest_generation()

# Important: The fitness values have to be in the same length and order as the neural networks are!
def get_next_generation(fitness):
    return selector.get_next_generation(fitness)