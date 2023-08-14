

# Neural network architecture (based on tensorflow)

INPUT_SIZE = 50
HIDDEN_LAYERS = 7
NODES_PER_LAYER = 500
OUTPUT_SIZE = 1
ACTIVATION_HIDDEN = 'relu'
ACTIVATION_OUTPUT = 'sigmoid'
INITIALIZER_SEED = 21

# Crossover settings

STANDARD_PROBABILITY = 1 # Children get weights directly from their parents, alternatively: mean weight of parents (1-p) times


# Mutation settings

MUTATION_RATE = 0.011 # Probability that mutation will occur on any given weight
MUTATION_RATE_BIAS = 0.021

# Population settings

POPULATION_SIZE = 60

# Selection criteria

TO_NEXT_GEN_UNCHANGED = 20
TO_NEXT_GEN_CROSSOVER = 35
TO_NEXT_GEN_CROSSOVER_MEAN = 0
TO_NEXT_GEN_NEW = 5

# Generational settings

GENERATIONS = 1000
SAVE_ALL_NEURAL_NETWORKS = False


# Meta data config

PROJECT_TITLE = "default"
META_DATA_FILENAME = 'metadata.txt'


