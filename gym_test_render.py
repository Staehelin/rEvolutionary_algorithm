import gym
import tensorflow as tf
import numpy as np
import time
import sys
import config
import pygame

import create_neural_networks
import meta_data_handler

meta_data = meta_data_handler.load_meta_data()
fitness = meta_data["generations_fitness_list"]
fitness_25 = fitness[::-1][0]
print(fitness_25)

max_index = fitness_25.index(max(fitness_25))
min_index = fitness_25.index(min(fitness_25))


# Load the best and worst models
best_model = tf.keras.models.load_model(f"./neural_nets/{config.PROJECT_TITLE}/latest_generation/neural_net_{max_index}")
#worst_model = tf.keras.models.load_model(f"./neural_nets/{config.PROJECT_TITLE}/latest_generation/neural_net_{min_index}")
worst_model = create_neural_networks.create_new_neural_network()

# Create the CartPole environment
env = gym.make('CartPole-v1',render_mode="human")

# Set seeds for reproducibility
seed_value = 102
np.random.seed(seed_value)

# Function to run simulation using a given model and visualize it
def run_simulation_with_render(model):
    total_reward = 0
    max_steps = 1000  # Maximum number of steps for the simulation

    observation, _  = env.reset()

    for step in range(max_steps):
        # Predict action using the model
        prediction = model.predict(np.array([observation]))
        score = prediction[0][0]
        action = 0 if score < 0.5 else 1

        # Perform the chosen action in the environment
        step_result = env.step(action)
        observation, reward, done, _ = step_result[:4]
        total_reward += reward

        # Render the environment to visualize the simulation
        env.render()
        time.sleep(0.001)

        if done:
            break

    return total_reward

# Run simulations using the best and worst models with visualization
worst_reward = run_simulation_with_render(worst_model)
best_reward = run_simulation_with_render(best_model)

# Print results
print(f"Worst model reward: {worst_reward}")
print(f"Best model reward: {best_reward}")

# Close the environment
env.close()
