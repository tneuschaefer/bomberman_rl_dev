from distutils import dist
import enum
import os
import pickle
from pydoc import cram
import random
from re import A

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        # ANGELINA
        self.q_table = []

        if os.path.isfile("q_table.csv"):
            with open("q_table.csv", "rb") as file:
                self.q_table = pickle.load(file)

        print("open q- table: size", len(self.q_table))

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            # ANGELINA

        if os.path.isfile("q_table.csv"):
            with open("q_table.csv", "rb") as file:
                self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # random_prob = .1
    # if self.train and random.random() < random_prob:
    #     self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
    #     return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model)

    if self.train and random.uniform(0, 1) < self.epsilon:
        action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])
        #print("Random action with epsilon")

    else:
        #action = np.random.choice(ACTIONS, p=[0, 0, 0, 0, 0, 1])
        action = get_action(self, game_state)

    self.logger.debug(
        f"at state {state_to_features(game_state)} executed action {action}")

    return action


def get_action(self, game_state):
    current_state = state_to_features(game_state)

    if current_state in self.q_table:
        action_index = np.argmax(self.q_table[current_state])
        action = ACTIONS[action_index]

    else:
        action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])
        print("state not in q:", current_state)

    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # # This is the dict before the game begins and after it ends
    # if game_state is None:
    #     return None

    # # For example, you could construct several channels of equal shape, ...
    # channels = []
    # channels.append(...)
    # # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # # and return them as a vector
    # return stacked_channels.reshape(-1)

    if game_state is None:
        return None

    field = game_state['field']
    coins = game_state['coins']
    own_x, own_y = game_state['self'][3]
    crates_i = np.where(field == 1)
    crates = list(zip(crates_i[0], crates_i[1]))

    # check if coins exist
    if len(coins) > 0:
        distance = np.zeros((len(coins), 2))
        for i, (x, y) in enumerate(coins):
            dist_x = own_x - x
            dist_y = own_y - y
            distance[i, 0] = dist_x
            distance[i, 1] = dist_y

        assert len(np.sum(distance, axis=1)) == len(coins)
        dist_closest_coin = distance[np.argmin(np.sum(distance ** 2, axis=1))]
        assert len(dist_closest_coin) == 2
    else:
        dist_closest_coin = [0, 0]

    # check if crates exist
    if len(crates) > 0:
        distance = np.zeros((len(crates), 2))
        for i, (x, y) in enumerate(crates):
            dist_x = own_x - x
            dist_y = own_y - y
            distance[i, 0] = dist_x
            distance[i, 1] = dist_y

        dist_closest_crate = distance[np.argmin(np.sum(distance ** 2, axis=1))]
        assert len(dist_closest_crate) == 2
    else:
        dist_closest_crate = [0, 0]

    # check if bomb is close

    environment = np.zeros(4)
    if field[own_x - 1, own_y] == 0:
        environment[0] = 1
    if field[own_x + 1, own_y] == 0:
        environment[1] = 1
    if field[own_x, own_y - 1] == 0:
        environment[2] = 1
    if field[own_x, own_y + 1] == 0:
        environment[3] = 1

    features = np.append(dist_closest_coin, environment)

    return tuple(features)
