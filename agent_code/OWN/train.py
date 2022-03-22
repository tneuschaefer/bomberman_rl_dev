from collections import namedtuple, deque

import os
import pickle
import csv
from typing import List

import numpy as np
import random
import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
VALID_ACTION = "VALID_ACTION"
MOVED_TO_COIN = "MOVED_TO_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
MOVED_IN_CYCLE = "MOVED_IN_CYCLE"
NOT_MOVED_IN_CYCLE = "NOT_MOVED_IN_CYCLE"

BOMB_WHEN_NO_COIN = "BOMB_WHEN_NO_COIN"
NOT_BOMB_WHEN_NO_COIN = "NOT_BOMB_WHEN_NO_COIN"
BOMB_IF_NO_COIN_AND_CLOSE_TO_CRATE = "BOMB_IF_NO_COIN + CLOSE_TO_CRATE"
NOT_BOMB_IF_NO_COIN_AND_CLOSE_TO_CRATE = "NOT BOMB_IF_NO_COIN + CLOSE_TO_CRATE"

NOT_BOMB_WHEN_COIN_EXISTS = "NOT_BOMB_WHEN_COIN_EXISTS"
BOMB_WHEN_COIN_EXISTS = "BOMB_WHEN_COIN_EXISTS"  # SUBJECT TO CHANGE

MOVED_TO_BOMB = 'MOVED_TO_BOMB'
MOVED_AWAY_FROM_BOMB = 'MOVED_AWAY_FROM_BOMB'


CLOSE_TO_CRATE = 'CLOSE_TO_CRATE'

ACTION_ROW_MAP = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4,
    'BOMB': 5
}

decay_rate = 0.005


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # ANGELINA
    self.alpha = 0.1
    self.gamma = 0.5
    self.epsilon = 0.4

    num_walls = 49
    # free cells robot can walk or a coin can be placed
    black_cells = 15 * 15 - num_walls
    num_states = 7 * 7 * 7 * 7
    num_actions = 6

    if not os.path.isfile("q_table.csv"):
        self.q_table = {(0, 0, 0, 0, 0, 0): [0, 0, 0, 0, 0, 0]}

    self.training_data = np.zeros((0, 81))

    self.all_epochs, self.all_penalties = [], []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # self.logger.debug(
    #     f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # # state_to_features is defined in callbacks.py

    update_q_table(self, new_game_state, self_action, old_game_state, events)


def update_q_table(self, new_game_state, self_action, old_game_state, events):
    new_state, old_state = get_states(new_game_state, old_game_state)
    all_events = add_auxilliary_rewards(self, old_state, new_state, events)

    self.transitions.append(Transition(
        old_state, self_action, new_state, reward_from_events(self, all_events)))

    new_q_value = get_new_q_value(self)
    self.q_table[old_state][ACTION_ROW_MAP[self_action]] = new_q_value


def get_states(new_game_state, old_game_state):
    new_state = state_to_features(new_game_state)

    if old_game_state is None:
        old_state = new_state
    else:
        old_state = state_to_features(old_game_state)

    return (new_state, old_state)


def get_new_q_value(self):
    old_state, self_action, new_state, rewards = self.transitions[-1]

    if old_state not in self.q_table:
        new_row = np.zeros(6)
        new_row[ACTION_ROW_MAP[self_action]] = rewards
        self.q_table.update({old_state: new_row})

    if new_state not in self.q_table:
        new_row = np.zeros(6)
        new_row[ACTION_ROW_MAP[self_action]] = rewards
        self.q_table.update({new_state: new_row})

    old_value = self.q_table[old_state][ACTION_ROW_MAP[self_action]]
    action_value = np.max(self.q_table[new_state])  # next_max

    new_value = old_value + self.alpha * \
        (rewards + self.gamma * action_value - old_value)

    return new_value


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # self.logger.debug(
    #     f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(
    #     last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    update_q_table(self, last_game_state, last_action, last_game_state, events)

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    with open('q_table.csv', 'wb') as csv_file:
        pickle.dump(self.q_table, csv_file, protocol=pickle.HIGHEST_PROTOCOL)

    print("Close q- table: size", len(self.q_table))

    # self.epsilon -= decay_rate
    # print("NEW EPSILON: ", self.epsilon)


def add_auxilliary_rewards(self, old_state, new_state, events):
    # reward for moving to nearest coin
    old_distance = np.sum(np.abs(old_state[:2]))
    new_distance = np.sum(np.abs(new_state[:2]))
    if old_distance < new_distance:
        events.append(MOVED_AWAY_FROM_COIN)
    else:
        events.append(MOVED_TO_COIN)

    # check agent moves in cycle
    if(len(self.transitions) > 2):
        st1 = self.transitions[1].next_state
        st2 = self.transitions[2].next_state
        if(new_state in [st1, st2]):
            events.append(MOVED_IN_CYCLE)
            print("in CYCLE: ", new_state)
        else:
            events.append(NOT_MOVED_IN_CYCLE)

    # add valid action
    if e.INVALID_ACTION not in events:
        events.append(VALID_ACTION)

    # drop bomb if no coins
    # if old_distance == 0:  # no coins in sight

    return events


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 50,
        e.KILLED_OPPONENT: 50,
        e.GOT_KILLED: -30,
        e.SURVIVED_ROUND: 30,
        e.INVALID_ACTION: -100,
        VALID_ACTION: 100,
        MOVED_TO_COIN: 5,
        MOVED_AWAY_FROM_COIN: -5,
        MOVED_IN_CYCLE: -5,
        NOT_MOVED_IN_CYCLE: 5,

        BOMB_WHEN_NO_COIN: 5,
        NOT_BOMB_WHEN_COIN_EXISTS: 5,
        NOT_BOMB_WHEN_NO_COIN: -5,
        BOMB_IF_NO_COIN_AND_CLOSE_TO_CRATE: 5,
        NOT_BOMB_IF_NO_COIN_AND_CLOSE_TO_CRATE: -5,

        MOVED_TO_BOMB: -5,
        MOVED_AWAY_FROM_BOMB: 5
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
