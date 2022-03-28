from collections import namedtuple, deque

import os
import pickle
import csv
from typing import List

import numpy as np
from pandas import DataFrame
import random
import events as e
from .callbacks import state_to_features, q_table_path, model_path, analytics_path, active_bomb_list
from scipy.spatial.distance import cityblock


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

SAFE_FROM_BOMB = 'SAFE_FROM_BOMB'
NOT_SAFE_FROM_BOMB = "NOT_SAFE_FROM_BOMB"

MOVED_TO_CRATE = "MOVED_TO_CRATE"
MOVED_AWAY_FROM_CRATE = "NOT_MOVE_TO_CRATE"

MEANINGFUL_BOMB_DROP = "MEANINGFUL_BOMB_DROP"
NOT_MEANINGFUL_BOMB_DROP = "NOT_MEANINGFUL_BOMB_DROP"
MOVED_TO_BOMB = 'MOVED_TO_BOMB'
MOVED_AWAY_FROM_BOMB = 'MOVED_AWAY_FROM_BOMB'
DIDNT_DROP_BOMB = 'DIDNT_DROP_BOMB'

MEANINGFUL_WAIT = 'MEANINGFUL_WAIT'
NOT_MEANINGFUL_WAIT = "NOT_MEANINGFUL_WAIT"

NOT_KILLED_SELF = "NOT_KILLED_SELF"


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
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # ANGELINA
    self.alpha = 0.8
    self.gamma = 0.9
    self.epsilon = 0.5
    if not os.path.isfile(q_table_path):
        self.q_table = {}

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

    update_q_table(self, new_game_state, self_action,
                   old_game_state, events, new_game_state['step'])
    dump_file(q_table_path, "wb", self.q_table)


def update_q_table(self, new_game_state, self_action, old_game_state, events, step):
    old_state, new_state = get_states(self, new_game_state, old_game_state)
    all_events = add_auxilliary_rewards(self, old_state, new_state, events)
    rewards = reward_from_events(self, all_events)

    self.logger.info(
        f"moving from {old_state} to {new_state}")

    # decay rewards by timestep
    # rewards = (0.99 * np.exp(-0.01 * new_game_state['step'])) * rewards
    rewards = rewards + potential_fct(new_state, old_state, step, self)
    self.logger.debug(f"New reward: {rewards}")

    self.transitions.append(Transition(
        old_state, self_action, new_state, rewards))

    new_q_value = get_new_q_value(self)
    self.q_table[old_state][ACTION_ROW_MAP[self_action]] = new_q_value

    self.logger.info(
        f"New q-value is {new_q_value} at state {old_state} for action {ACTION_ROW_MAP[self_action]}")

    self.logger.info("")


def potential_fct(new_state, old_state, step, self):
    """Compute Fi for the reward shapping
    """
    Fi_old = fi_reward_shaping(old_state, step-1)
    Fi_new = fi_reward_shaping(new_state, step)

    F = -1 * step
    self.logger.debug(f"F equals {F}")
    return F


def potential_fct_old(new_state, old_state, step, self):
    """Compute Fi for the reward shapping
    """
    Fi_old = fi_reward_shaping(old_state, step-1)
    Fi_new = fi_reward_shaping(new_state, step)

    F = self.gamma*Fi_new - Fi_old
    self.logger.debug(f"F equals {F}= ({Fi_new} - {Fi_old})")
    return F


def fi_reward_shaping(state, step):
    if np.sum(np.abs(state[8:10])) != 0:
        GOAL = list((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0))
    else:  # no bomb exists
        GOAL = list((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    if np.sum(np.abs(state[6:8])) != 0:
        GOAL1 = list((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        GOAL2 = list((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        if cityblock(GOAL1, state, w=[1, 1, 0, 0, 0, 0, 3, 3, 4, 4]) < cityblock(GOAL2, state, w=[1, 1, 0, 0, 0, 0, 3, 3, 4, 4]):
            GOAL = GOAL1
        else:
            GOAL = GOAL2

    if np.sum(np.abs(state[6:8])) == 1:
        GOAL = list((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0))

    dist = cityblock(GOAL, state, w=[1, 1, 0, 0, 0, 0, 3, 3, 4, 4])

    Fi = dist * 1/4 * step
    return Fi


def get_states(self, new_game_state, old_game_state):
    new_state = state_to_features(new_game_state)

    if old_game_state is None:
        old_state = new_state
        self.logger.warn("old state NONE")
    else:
        old_state = state_to_features(old_game_state)

    return (old_state, new_state)


def get_new_q_value(self):
    old_state, self_action, new_state, rewards = self.transitions[-1]

    if old_state not in self.q_table:
        # new_row = q_t_initialization(old_state)
        new_row = np.zeros(6)
        new_row[ACTION_ROW_MAP[self_action]] = rewards
        self.q_table.update({old_state: new_row})

    if new_state not in self.q_table:
        # new_row = q_t_initialization(new_state)
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

    # Store the model
    update_q_table(self, last_game_state, last_action,
                   last_game_state, events, last_game_state['step'])
    # delete old bomb data
    active_bomb_list[:] = []

    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    dump_file(model_path, "wb", self.model)
    dump_file(q_table_path, "wb", self.q_table)

    score = last_game_state['self'][1]
    name = "scores/" + q_table_path.replace("csv", "txt")
    dump_analytics(name, score)


def dump_file(path: str, mode: str, object):
    with open(path, mode) as file:
        pickle.dump(object, file)


def dump_analytics(file_name, score):
    path = analytics_path + file_name
    df = DataFrame(data=[score], columns=['score'])
    df.to_csv(path, mode='a', header=not os.path.exists(path))


def add_auxilliary_rewards(self, old_state, new_state, events):

    # COINS
    add_coin_rewards(old_state, new_state, events)

    # CRATES
    add_crate_rewards(old_state, new_state, events)

    # BOMBS
    add_bomb_rewards(old_state, new_state, events)

    # rewards to incentivize good playing behavior
    add_general_rewards(self, new_state, events)

    return events


def add_general_rewards(self, new_state, events):
    # add valid action
    if e.INVALID_ACTION not in events:
        events.append(VALID_ACTION)

     # check agent moves in cycle
    if(len(self.transitions) > 2):
        st1 = self.transitions[1].next_state
        st2 = self.transitions[2].next_state
        if(new_state in [st1, st2]):
            events.append(MOVED_IN_CYCLE)
            # print("in CYCLE: ", new_state)
        else:
            events.append(NOT_MOVED_IN_CYCLE)

    # agent survived bomb
    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events:
        events.append(NOT_KILLED_SELF)

    if e.WAITED in events and MEANINGFUL_WAIT not in events:
        events.append(NOT_MEANINGFUL_WAIT)

    if e.BOMB_DROPPED in events and MEANINGFUL_BOMB_DROP not in events:
        events.append(NOT_MEANINGFUL_BOMB_DROP)


def add_coin_rewards(old_state, new_state, events):
    # COINS
    # distances correspond to manhattan distance
    old_distance_coin = np.sum(np.abs(old_state[:2]))
    new_distance_coin = np.sum(np.abs(new_state[:2]))

    # reward for moving to nearest coin if one exists
    if old_distance_coin != 0:
        if old_distance_coin <= new_distance_coin:
            if e.COIN_COLLECTED not in events:
                events.append(MOVED_AWAY_FROM_COIN)
        else:
            events.append(MOVED_TO_COIN)


def add_crate_rewards(old_state, new_state, events):
    left, right, up, down = old_state[2:6]
    old_distance_crate = np.sum(np.abs(old_state[6:8]))
    old_crate_x, old_crate_y = np.abs(old_state[6:8])
    new_distance_crate = np.sum(np.abs(new_state[6:8]))
    new_crate_x, new_crate_y = np.abs(new_state[6:8])

    old_distance_coin = np.sum(np.abs(old_state[:2]))
    new_distance_bomb = np.sum(np.abs(new_state[8:10]))

    # # move to crate when no coins visible
    # # no coins in sight + at least one crate exists
    if old_distance_coin == 0 and old_distance_crate != 0:
        # didnt drop bomb
        if new_distance_bomb == 0:
            if old_distance_crate <= 3 and new_distance_crate > 3:
                events.append(DIDNT_DROP_BOMB)
            if old_distance_crate > new_distance_crate:
                events.append(MOVED_TO_CRATE)
            # agent moved away from crate + a bomb was not the reason
            else:
                events.append(MOVED_AWAY_FROM_CRATE)

    if e.BOMB_DROPPED in events:
        # placing bomb near crate is good
        if old_crate_x <= 3 and old_crate_y == 0 or old_crate_x == 0 and old_crate_y <= 3:
            if (old_crate_x == 0 and up == 0 or down == 0) or (old_crate_y == 0 and right == 0 or left == 0):
                events.append(MEANINGFUL_BOMB_DROP)
        else:
            # too far from crate to drop a bomb
            events.append(NOT_MEANINGFUL_BOMB_DROP)


def add_bomb_rewards(old_state, new_state, events):
    old_distance_bomb = np.sum(np.abs(old_state[8:10]))
    new_distance_bomb = np.sum(np.abs(new_state[8:10]))

    old_bomb_dist_x, old_bomb_dist_y = np.abs(old_state[8:10])
    new_bomb_dist_x, new_bomb_dist_y = np.abs(new_state[8:10])

    # # if a bomb exists
    if old_distance_bomb != 0:

        # case agent was standing on bomb
        if old_distance_bomb == 200:

            if new_distance_bomb != 200:
                events.append(MOVED_AWAY_FROM_BOMB)
                # events.append(NOT_SAFE_FROM_BOMB)

            # agent was standing on bomb + is still standing
            if new_distance_bomb == 200:
                events.append(NOT_SAFE_FROM_BOMB)
                # choosing to drop a bomb when standing on one is an invalid action
                if e.INVALID_ACTION in events and NOT_MEANINGFUL_BOMB_DROP not in events:
                    events.append(NOT_MEANINGFUL_BOMB_DROP)
                else:
                    events.append(NOT_MEANINGFUL_WAIT)

        #     # move away from dropped bomb
        if old_distance_bomb < new_distance_bomb:
            # outside dangerous radius of bomb
            if (new_bomb_dist_x > 3 and new_bomb_dist_y == 0) or (new_bomb_dist_y > 3 and new_bomb_dist_x == 0):
                events.append(SAFE_FROM_BOMB)
            else:
                events.append(NOT_SAFE_FROM_BOMB)
            if (old_bomb_dist_x <= 3 and new_bomb_dist_y == 0) or (old_bomb_dist_y <= 3 and new_bomb_dist_x == 0):
                events.append(MOVED_AWAY_FROM_BOMB)

            # moved closer to bomb but wasnt standing on one
        if old_distance_bomb >= new_distance_bomb and old_distance_bomb != 200:
            # inside dangerous radius of bomb
            if (new_bomb_dist_x <= 5 and new_bomb_dist_y == 0) or (new_bomb_dist_y <= 5 and new_bomb_dist_x == 0):
                events.append(MOVED_TO_BOMB)
                events.append(NOT_SAFE_FROM_BOMB)

    #     # wait is ok if agent in safe position from a dropped bomb
    # ANGELINA removed old_bomb_dist_x == 0 or old_bomb_dist_y == 0 and
    # this safe from bomb not working well yet !!!!!!
        if new_bomb_dist_x != 0 and new_bomb_dist_y != 0 and old_distance_bomb != 200:
            if e.WAITED:
                events.append(MEANINGFUL_WAIT)
            events.append(SAFE_FROM_BOMB)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        # GENERAL
        VALID_ACTION: 0,
        e.INVALID_ACTION: -2000,
        MOVED_IN_CYCLE: -2000,
        NOT_MOVED_IN_CYCLE: 0,

        # COINS
        MOVED_TO_COIN: 50,
        MOVED_AWAY_FROM_COIN: -5,
        e.COIN_COLLECTED: 1000,
        e.COIN_FOUND: 20,

        # CRATES
        e.CRATE_DESTROYED: 50,
        DIDNT_DROP_BOMB: -60,
        MOVED_TO_CRATE: 100,
        MOVED_AWAY_FROM_CRATE: -50,

        # BOMBS
        MOVED_TO_BOMB: -2000,
        MOVED_AWAY_FROM_BOMB: 200,
        MEANINGFUL_WAIT: 5,
        NOT_MEANINGFUL_WAIT: -30,
        SAFE_FROM_BOMB: 200,
        NOT_SAFE_FROM_BOMB: -200,
        MEANINGFUL_BOMB_DROP: 200,
        NOT_MEANINGFUL_BOMB_DROP: -400,

        # LIFE
        e.KILLED_SELF: -2000,
        NOT_KILLED_SELF: 0,
        e.KILLED_OPPONENT: 1000,
        e.OPPONENT_ELIMINATED: 50,
        e.GOT_KILLED: -2000,
        e.SURVIVED_ROUND: 0,
    }
    # remove duplicate events
    events = list(set(events))

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
