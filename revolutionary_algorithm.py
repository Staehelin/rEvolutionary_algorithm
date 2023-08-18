import config
import meta_data_handler
import os
import selector



def initialize():
    if (os.path.isfile(config.META_DATA_FILENAME)):
        meta_data_handler.load_meta_data()
    else:
        meta_data_handler.save_meta_data()

def get_starting_generation():
    if(meta_data_handler.meta_data == None or meta_data_handler.meta_data['generations_list'] == []):
        print("Initialize completely new random networks for the first generation.")
        return selector.get_first_generation()
    else:
        print("Loaded saved networks for the first generation.")
        return selector.load_latest_generation()

# Important: The fitness values have to be the same length and order as the neural networks are!
def get_next_generation(fitness):
    return selector.get_next_generation(fitness)
