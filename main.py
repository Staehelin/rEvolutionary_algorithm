import revolutionary_algorithm
import selector
import numpy as np
import revolutionary_algorithm as ra



def print_hi(name):
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('Thanks for using this rEvolutionary algorithm. Made by Lukas Staehelin')


ra.initialize()
ra.get_starting_generation()
ra.get_next_generation(list(range(21)))
ra.get_next_generation(list(range(100, 121)))

