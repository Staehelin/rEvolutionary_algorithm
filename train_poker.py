import math

import config as ra_config
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.api.game import setup_config, start_poker
import sys
import numpy as np

import create_neural_networks
import meta_data_handler
import revolutionary_algorithm as ra
import tensorflow as tf

streets_encoded = {
    "flop": 0,
    "turn": 1,
    "river": 2
}

cards_encoded = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12
}

suits_encoding = {
    "D": 0,
    "H": 1,
    "S": 2,
    "C": 3
}

# Take card and returns numerical representation of value (e.g. 2 -> 0, Ace -> 12)
def card_value_encoded(card):
    value = card[1]
    value = cards_encoded[value]
    return value

#TODO: Write test for game_state_to_input_vector
#Manually create one input vector and gamestate, and test if it translated correctly
def game_state_to_input_vector(hole_card, round_state,uuid):
    """
    :param hole_card:
    :param round_state:
    :param valid_actions:

    :Logic of inputvector
    2x 13 holecards values 2-A
    5x 13 Flops,Turns,Rivers card values 2-A
    2x 5 Community card same suits as hole card 1 and 2 ?
    3x 5 Community cards not same suits as holecard 1 or 2 (can be a max of 3, if holecards are suited)
    1x Potsize / SB
    6x is dealer
    6x isNextPlayer
    6x Participating in current hand
    6x stacksize / sb (FOR FUTURE: 0 if empty seat!!)
    6x money currently invested already (by calling, betting, raising) / sb
    1x highest bet/raise   /sb


    :return:
    """



    offset = 0
    input_vector = np.zeros(147)
    suits = ["H","D","C","S"]
    # Convert hole card value and save suits of hole cards
    card1 = hole_card[0]
    card2 = hole_card[1]
    suits.remove(card1[0])
    if(card1[0] != card2[0]):
        suits.remove(card2[0])
    suit1 = suits_encoding[card1[0]]
    suit2 = suits_encoding[card2[0]]

    input_vector[card_value_encoded(card1)] = 1
    offset += 13
    input_vector[card_value_encoded(card2) + offset] = 1
    offset += 13


    for i, community_card in enumerate(round_state['community_card']):
        # Add face value to input vector
        input_vector[card_value_encoded(community_card) + 13 * i + offset] = 1

        # Handle suits
        # Either add 2x 5 arrays if each community card has same suits as holecards
        # Or see if rest of board has same suits that are not the suits of holecards
        if suits_encoding[community_card[0]] == suit1:
            input_vector[13 * 5 + offset + i] = 1
        if suits_encoding[community_card[0]] == suit2:
            input_vector[13 * 5 + offset + 5 + i] = 1
        for j in range(len(suits)):
            #Community card has the same color as a non holecard color
            if community_card[0] == suits[j]:
                input_vector[13 * 5 + offset + 5 * 2 + i + 5 * j] = 1


    # Positional information and pot information
    sb = round_state['small_blind_amount'] * 100
    offset_cards = 13 * 5 + 2 * 13 + 5 * 2 + 5 * 3
    input_vector[offset_cards + round_state['dealer_btn']] = 1

    #next player is equal to current player position
    input_vector[offset_cards + 6 + round_state['next_player']] = 1

    participating_uuids = []
    for i, seat in enumerate(round_state['seats']):
        if(seat['state'] == 'participating'):
            input_vector[offset_cards + 2 * 6 + i] = 1
            participating_uuids.append(seat['uuid'])
        # Stacks of each player
        input_vector[offset_cards + 3 * 6 + i] = seat['stack'] / sb

    # Calculate the money currently invested thats not in pot (calls, bets, and raises of current round)
    #Participating uuids are ordered how theyre supposed to (i believe), thus i will be the offset for amount investead already
    actions_streets = round_state['action_histories']
    streets = ['preflop','flop','turn','river']
    for i, uuid in enumerate(participating_uuids):
        for street in streets:
            if(street in actions_streets):
                actions = actions_streets[street]
                for action in actions:
                    if(action['uuid'] == uuid):
                        input_vector[offset_cards + (3+1) * 6 + i] = action['amount'] / sb

    #TODO: implement side pots
    # current potsize (side pots not implemented yet)
    input_vector[offset_cards + 5 * 6] = round_state['pot']['main']['amount'] / sb

    print(input_vector)

    return input_vector


class FishPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self,neural_net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_net = neural_net

        # Rest of your FishPlayer class implementation


    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):

        uuid = self.uuid

        vector = game_state_to_input_vector(hole_card, round_state, uuid)
        vector = np.reshape(vector, (1, 147))

        # Insert nn here
        prediction = self.neural_net.predict(vector)
        score = prediction[0][0]
        nn_action = score * round_state['pot']['main']['amount'] * 2

        sb_100 = round_state['small_blind_amount'] * 100
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        if(nn_action >= valid_actions[2]['amount']['min'] and valid_actions[2]['amount']['min'] != -1):
            action = 'raise'
            if(nn_action > valid_actions[2]['amount']['max']):
                amount = valid_actions[2]['amount']['max']
            else:
                amount = math.floor(nn_action)
        elif(nn_action >= valid_actions[1]['amount']):
            action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        else:
            action, amount = 'fold', 0


        print(f"{self.uuid} did {action} for {amount} with {hole_card}")


        return action, amount   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def test_latest_gen(project_title=ra_config.PROJECT_TITLE):
    ra.initialize()
    meta_data = meta_data_handler.load_meta_data()
    fitness = meta_data["generations_fitness_list"]
    print(fitness)
    #fitness_25 = fitness[::-1][0]

    #max_index = fitness_25.index(max(fitness_25))
    max_index = 0
    best_model = tf.keras.models.load_model(
        f"./neural_nets/{project_title}/latest_generation/neural_net_{max_index}")
    config = setup_config(max_round=21, initial_stack=115, small_blind_amount=1)
    for j in range(6):
        if(j<1):
            config.register_player(name=f"p{j}", algorithm=FishPlayer(best_model))
        else:
            # Save network false, otherwise it'll override the saved networks
            config.register_player(name=f"p{j}", algorithm=FishPlayer(create_neural_networks.create_new_neural_network(save_network=False)))

    game_result = start_poker(config, verbose=1)
    fitness = []
    for result in game_result['players']:
        fitness.append(result['stack'])

    print(fitness)

    sys.exit(1)


test_latest_gen("poker")
#sys.exit(1)

ra.initialize()
population = ra.get_starting_generation()
for i in range(ra_config.GENERATIONS):
    fitness = []
    config = setup_config(max_round=21, initial_stack=115, small_blind_amount=1)
    for j, pop in enumerate(population):
        config.register_player(name=f"p{j}", algorithm=FishPlayer(pop))

        if(j != 0 and j%6 == 5):
            game_result = start_poker(config, verbose=0)
            for result in game_result['players']:
                fitness.append(result['stack'])
            config = setup_config(max_round=21, initial_stack=115, small_blind_amount=1)

    population = ra.get_next_generation(fitness)

