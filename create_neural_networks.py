import tensorflow as tf
import numpy as np
import config


# Create a new nerual network
def create_new_neural_network(input_size=config.INPUT_SIZE, hidden_layers=config.HIDDEN_LAYERS, nodes_per_layer=config.NODES_PER_LAYER, output_size=config.OUTPUT_SIZE, activation_hidden=config.ACTIVATION_HIDDEN, activation_output=config.ACTIVATION_OUTPUT):
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=(input_size,)))

    # Hidden layers
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(nodes_per_layer, activation=activation_hidden))

    # Output layer
    model.add(tf.keras.layers.Dense(output_size, activation=activation_output))

    return model

# Create new child neural network ouf of the weights and biases of both parents according to their fitness evaluations
def crossover_neural_network(parent1, parent2, fitness1, fitness2, standard=True):
    new_neural_net = create_new_neural_network()
    layers_new = new_neural_net.layers

    # Create mutation values for mutations (HeUniform for ReLu, GlorotUniform for sigmoid, <- unconfirmed information)
    initializer = tf.keras.initializers.HeUniform(seed=config.INITIALIZER_SEED)


    if(standard):
        p_value = fitness1/(fitness1+fitness2)

        layers1 = parent1.layers
        layers2 = parent2.layers

        for layer1, layer2, layer_new in zip(layers1, layers2, layers_new):
            w1, b1 = layer1.get_weights()
            w2, b2 = layer2.get_weights()

            # Create masks for the current layer
            layer_size = len(layer1.get_weights())
            random_numbers_w = np.random.rand(*w1.shape)
            random_numbers_b = np.random.rand(*b1.shape)
            random_numbers_mutate_w = np.random.rand(*w1.shape)
            random_numbers_mutate_b = np.random.rand(*b1.shape)
            mask_w = random_numbers_w <= p_value
            mask_b = random_numbers_b <= p_value
            mask_mutate_w = random_numbers_mutate_w <= config.MUTATION_RATE
            mask_mutate_b = random_numbers_mutate_b <= config.MUTATION_RATE_BIAS

            w_updated = np.where(mask_w, w1, w2)
            b_updated = np.where(mask_b, b1, b2)

            w_mutation_values = initializer(shape=(w_updated.shape))
            b_mutation_values = initializer(shape=(b_updated.shape))
            w_mutated = np.where(mask_mutate_w, w_mutation_values, w_updated)
            b_mutated = np.where(mask_mutate_b, b_mutation_values, b_updated)


            layer_new.set_weights([np.array(w_mutated), np.array(b_mutated)])
    else:
        print("Not implemented yet, set \"TO_NEXT_GEN_CROSSOVER_MEAN = 0\". This returned just a newly randomly initialized neural network instead.")
        pass




    return new_neural_net








