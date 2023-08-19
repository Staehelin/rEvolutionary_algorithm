import numpy as np

from train_poker import game_state_to_input_vector

hole_cards = ['HK', 'CA']
community_cards = ['HQ', 'HJ', 'CA', 'S2', 'D3']

def test_game_state_to_input_vestor(hole_card, community_cards):
    round_state = {'community_card': community_cards}
    expected_output = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                                0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0.,0.,0.,0.,0.])

    print(expected_output.size)

    input_vector = game_state_to_input_vector(hole_card, round_state)

    # Count the occurrences of 1
    count_of_ones_expected = np.count_nonzero(expected_output == 1)
    count_of_ones_input = np.count_nonzero(input_vector == 1)

    print("Count of ones in expected output:", count_of_ones_expected)
    print("Count of ones in input vector:", count_of_ones_input)

    # Find the indices of all occurrences of 1
    indices_expected = np.where(expected_output == 1)[0]
    indices_input = np.where(input_vector == 1)[0]

    print("Indices of ones in expected output:", indices_expected)
    print("Indices of ones in input vector:", indices_input)


test_game_state_to_input_vestor(hole_cards, community_cards)