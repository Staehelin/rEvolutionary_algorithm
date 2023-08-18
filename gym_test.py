import gym

import config
import revolutionary_algorithm as ra
import random
import numpy as np

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Set seeds for reproducibility
seed_value = 55
env.action_space.seed(seed_value)

# Configuration
POPULATION_SIZE = config.POPULATION_SIZE
NUM_GENERATIONS = config.GENERATIONS
max_steps = 500

# Get the first generation of models
models = ra.get_starting_generation()

for generation in range(NUM_GENERATIONS):
    print(f"Training Generation {generation + 1}")

    # Evaluate models in the current generation
    fitness_array = []
    for model in models:
        total_reward = 0
        observation, _ = env.reset()
        print(observation)

        for step in range(max_steps):

            prediction = model.predict(np.array([observation]))
            score = prediction[0][0]
            action_discrete = 0 if score < 0.5 else 1
             # Use your model's action selection method here

            step_result = env.step(action_discrete)

            observation, reward, done, _ = step_result[:4]
            total_reward += reward

            if done:
                break

        fitness_array.append(total_reward)

    # Get the next generation of models
    print(max(fitness_array))
    models = ra.get_next_generation(fitness_array)

    env.action_space.seed(seed_value + generation)

# Training completed, close the environment
env.close()

