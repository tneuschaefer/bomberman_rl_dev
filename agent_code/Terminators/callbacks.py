from distutils import dist
import enum
from json import load
import os
import pickle
from pydoc import cram
import random
from re import A
import weakref

import numpy as np
from random import shuffle
from scipy.spatial import distance
from scipy.spatial.distance import cityblock, cdist
import time


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# a list of active bombs to keep track of bombs that have exploded and are dangerous for one more time step
active_bomb_list = []

# 2-test-q-initialization.csv
q_table_path = "data/q_table-final.csv"
# "q_table-task2.csv"
model_path = "data/my-saved-model.pt"
analytics_path = "analytics/"


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
        self.q_table = load_file(q_table_path, "rb")

    else:
        self.logger.info("Loading model from saved state.")
        self.model = load_file(model_path, "rb")
        self.q_table = load_file(q_table_path, "rb")


def load_file(path: str, mode: str):
    if os.path.isfile(path):
        with open(path, mode) as file:
            return pickle.load(file)
    return []


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
    start_time = time.time()

    if self.train and random.uniform(0, 1) < self.epsilon:
        current_state = state_to_features(game_state)

        if current_state in self.q_table and 0 in self.q_table[current_state]:
            # choose action that hasnt been tried out yet
            action_index = np.where(self.q_table[current_state] == 0)[0][0]
            action = ACTIONS[action_index]
            #print(f"choose 0 action at state {current_state} action_i {action_index}, also action = {action}")
        else:
            action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

        self.logger.info(f"Agent standing on {game_state['self'][3]}")
        self.logger.info(f"Bombs standing on {game_state['bombs']}")
        self.logger.info(
            f"RANDOM CHOICE at state {state_to_features(game_state)}, action {action}")
    else:
        action = get_action(self, game_state)
    end_time = time.time()
    self.logger.debug(f"Time to decide: {end_time - start_time} sec.")
    return action


def get_action(self, game_state):
    current_state = state_to_features(game_state)

    if self.train and current_state not in self.q_table:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    if not self.train and current_state not in self.q_table:
        current_state = get_most_similar_state(self, current_state)
        self.logger.warning(
            f"unseen state: {state_to_features(game_state)} ,found most similar state: {current_state}")

    action_index = np.argmax(self.q_table[current_state])
    action = ACTIONS[action_index]

    self.logger.debug(
        f"at state {current_state} executed action {action}")
    return action


def get_most_similar_state(self, current_state):

    keys = list(self.q_table.keys())
    # distances = []
    # for key in keys:
    #     # for mannhatan distance cityblock(key, current_state, w=[1, 1, 4, 4, 4, 4, 2, 2, 3, 3])
    #     distances.append(distance.euclidean(key, current_state, w=[
    #                      3, 3, 2, 2, 2, 2, 1, 1, 4, 4]))

    # return keys[np.argmin(distances)]
    dist = cdist(keys, [current_state, current_state], metric='euclidean')
    return keys[np.argmin(dist, axis=1)[0]]


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
    if game_state is None:
        return None

    own_x, own_y = game_state['self'][3]
    field = game_state['field']
    crates_i = np.where(field == 1)
    crates = list(zip(crates_i[0], crates_i[1]))
    opponents = game_state['others']
    opponents_coordinates = [xy for (n, s, b, xy) in opponents]

    # coin
    field = game_state['field']
    field = (field == 0)
    closest_reachable_obj_coordinates = look_for_targets(
        field, start=game_state['self'][3], targets=game_state['coins'], logger=None)

    #print("reachable coins coordinates: ", closest_reachable_obj_coordinates)
    # Manhattan distance
    dist_closest_coin = get_nearest_obj(
        closest_reachable_obj_coordinates, own_x, own_y)

    # crate or opponent
    # Manhattan distance
    dist_closest_bombable = get_nearest_bombable_object(
        crates, opponents, own_x, own_y)

    # bomb
    # Manhattan distance
    dist_closest_bomb = get_nearest_bomb(game_state['bombs'], own_x, own_y)

    # environment to move
    # Manhattan distance
    environment = get_environment(
        field, opponents_coordinates, game_state['bombs'], own_x, own_y)

    features = np.concatenate(
        (dist_closest_coin, environment, dist_closest_bombable, dist_closest_bomb))

    return tuple(features)


def get_environment(field, opponents, bombs, own_x, own_y):
    for x, y in opponents:
        field[x, y] = -1

    for (x, y), t in bombs:
        field[x, y] = -1

    environment = np.zeros(4)
    if field[own_x - 1, own_y] == 0:
        environment[0] = 1
    if field[own_x + 1, own_y] == 0:
        environment[1] = 1
    if field[own_x, own_y - 1] == 0:
        environment[2] = 1
    if field[own_x, own_y + 1] == 0:
        environment[3] = 1

    return environment


def get_nearest_obj(objects: list, own_x, own_y):
    if len(objects) > 0:
        distance = np.zeros((len(objects), 2))
        for i, (x, y) in enumerate(objects):
            dist_x = own_x - x
            dist_y = own_y - y
            distance[i, 0] = dist_x
            distance[i, 1] = dist_y

        assert len(np.sum(distance, axis=1)) == len(objects)
        dist_closest_obj = distance[np.argmin(np.sum(distance ** 2, axis=1))]
        assert len(dist_closest_obj) == 2
    else:
        dist_closest_obj = np.array([0, 0])

    return dist_closest_obj


def get_nearest_bombable_object(crates: list, opponents: list, own_x, own_y):
    # use crates if crates still exist
    dist_closest_bombable = get_nearest_obj(crates, own_x, own_y)

    # if no crates exist use distance to nearest opponent
    if np.all(dist_closest_bombable == 0):
        opponents_coordinates = [xy for (n, s, b, xy) in opponents]
        dist_closest_bombable = get_nearest_obj(
            opponents_coordinates, own_x, own_y)

    return dist_closest_bombable


def get_nearest_bomb(bombs: list, own_x, own_y):
    """AI is creating summary for get_nearest_bomb

    Args:
        bombs (list): [description]
        own_x ([type]): [description]
        own_y ([type]): [description]

    Returns:
        [type]: [description]
    """
    bomb_coordinates = bombs
    # return bomb coordinates if bombs exists
    if len(bombs) != 0:
        bomb_coordinates = [xy for (xy, t) in bombs]

    # standard case
    dist_nearest_bomb = get_nearest_obj(bomb_coordinates, own_x, own_y)

    # special case when agent is on bomb
    if len(bombs) > 0 and np.all(dist_nearest_bomb == 0):
        # add bomb twice to capture it in old and new state
        active_bomb_list.append(bombs)
        active_bomb_list.append(bombs)
        dist_nearest_bomb = [100, 100]

    # special case when bomb exploded and is dangerous for one more step, but not present in game_state
    if len(bombs) == 0 and len(active_bomb_list) != 0:
        bomb = [xy for (xy, t) in active_bomb_list.pop()]
        dist_nearest_bomb = get_nearest_obj(bomb, own_x, own_y)
        # delete old bomb data
        #active_bomb_list[:] = []

    return dist_nearest_bomb


# BREATH FIRST SEARCH

def look_for_targets(free_space, start, targets, logger):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of the closest reachable target
    """
    if len(targets) == 0:
        return []

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y),
                                           (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    #logger.warning(f'Suitable target found at {best}')

    return [best]
