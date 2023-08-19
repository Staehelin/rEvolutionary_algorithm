import config
import create_neural_networks
import numpy as np
import tensorflow as tf

import meta_data_handler

current_generation = []

def set_current_generation(gen):
    global current_generation
    current_generation = gen
def get_cumulative_sum(sorted_values):
    # 0 required, so that choose_parent_index works properly (starting at index 1)
    cumulative_probablities = [0]
    cumulative_sum = 0

    sum_all_values = sum(sorted_values)

    if(sum_all_values == 0):
        return np.zeros(len(sorted_values))

    sorted_values_probablities = [sorted_value / sum_all_values for sorted_value in sorted_values]

    for probability in sorted_values_probablities:
        cumulative_sum += probability
        cumulative_probablities.append(cumulative_sum)

    return cumulative_probablities

def get_parent_index(sorted_values):
    # Returns random value from [0,1)
    random_value = np.random.random()
    becoming_a_parent_probabilities = get_cumulative_sum(sorted_values)

    for i in range(1, len(becoming_a_parent_probabilities)):
        if(becoming_a_parent_probabilities[i-1] <= random_value < becoming_a_parent_probabilities[i]):
            return i-1

    print("ERROR: Error in get_parent_index. Returned default value 0 instead.")
    return 0


def load_latest_generation():
    current_gen = []
    for i in range(config.POPULATION_SIZE):
        model = tf.keras.models.load_model(f"./neural_nets/{config.PROJECT_TITLE}/latest_generation/neural_net_{i}")
        model.id = -1
        current_gen.append(model)


    set_current_generation(current_gen)

    return current_gen

def get_first_generation():
    gen = []
    for _ in range(config.POPULATION_SIZE):
        gen.append(create_neural_networks.create_new_neural_network(save_network=True))

    meta_data_handler.meta_data['generations_list'].append(list(range(config.POPULATION_SIZE)))
    meta_data_handler.save_meta_data()

    set_current_generation(gen)

    return gen


def get_next_generation(current_generation_fitness):
    global current_generation

    # Updates generation value to current generation value
    create_neural_networks.update_current_generation()

    next_gen = []

    sorted_list = sorted(zip(current_generation_fitness, current_generation),key=lambda x: x[0], reverse=True)

    sorted_values, sorted_neural_nets = zip(*sorted_list)

    # Add TO_NEXT_GEN_UNCHANGED-many neural nets to next gen unchanged
    next_gen.extend(sorted_neural_nets[:config.TO_NEXT_GEN_UNCHANGED])

    # Add TO_NEXT_GEN_CROSSOVER-many crossed over neural nets to next gen

    for i in range(config.TO_NEXT_GEN_CROSSOVER):
        p1_index = get_parent_index(sorted_values)
        p2_index = get_parent_index(sorted_values)
        next_gen.append(create_neural_networks.crossover_neural_network(sorted_neural_nets[p1_index],
                                                                        sorted_neural_nets[p2_index],
                                                                        sorted_values[p1_index],
                                                                        sorted_values[p2_index]))

    # Add completely new networks to generation

    for i in range(config.TO_NEXT_GEN_NEW):
        next_gen.append(create_neural_networks.create_new_neural_network())


    next_gen_ids = [nn.id for nn in next_gen]
    meta_data_handler.meta_data['generations_list'].append(next_gen_ids)
    print(current_generation_fitness)
    meta_data_handler.meta_data['generations_fitness_list'].append(current_generation_fitness)
    meta_data_handler.meta_data["average_fitness_list"].append(sum(current_generation_fitness) / config.POPULATION_SIZE)
    meta_data_handler.save_meta_data()

    set_current_generation(next_gen)

    return next_gen

