import os
import agent_code.OWN.train as train
import time
import pickle
from multiprocessing import Pool
import numpy as np


def main():
    q_path = get_q_name()

    apsi_va = np.arange(0.5, 0.01, -1/500)

    start_time = time.time()

    exploration_decay_rate = 0.01

    for e in apsi_va:
        print("current epsilon: ", e)

        write_to_file(e)

        for _ in range(3):
            games = range(2)
            pool = Pool()
            pool.map(multi_process, games)
        print_q_table_info(q_path)

    end_time = time.time()
    print(f"Finished after {end_time - start_time:.2f} seconds ")
    print()
    print()


def get_q_name():
    """Get filename of q_table being trained.

    Returns
    -------
    [type]
        [description]
    """
    path = "/Users/angelinabasova/Library/CloudStorage/OneDrive-bwedu/bomberman_rl_dev/agent_code/OWN/callbacks.py"

    with open(path) as f:
        lines = f.readlines()

    q_path_i = 0
    for i, line in enumerate(lines):
        if "q_table_path = " in line:
            q_path_i = i
            break

    q_line = lines[q_path_i]
    q_line = q_line.replace("q_table_path = ", "")
    q_line = q_line.strip()
    return q_line


def all_actions_explored(file: str):
    """Check whether every action for evert state in the q table has been tried out. This method is only used when epsilon is 1 to ensure full exploration of the states. 

    Parameters
    ----------
    file : str
        [description]

    Returns
    -------
    [type]
        [description]
    """
    path = "/Users/angelinabasova/Library/CloudStorage/OneDrive-bwedu/bomberman_rl_dev/agent_code/OWN/" + file
    path = path.replace("\"", "")
    with open(path, "rb") as file:
        q_table = pickle.load(file)
    val = list(q_table.values())
    val = np.array(val)
    return (0 not in val)


def multi_process(game: int):
    os.system(
        "python main.py play --no-gui --agents OWN --train 1 --scenario coin-heaven --n-rounds 1000")
    # remove seed when training complete


def write_to_file(epsilon):
    """Overwrite the epsilon value in the train.py file in order to train with the new overwritten value.

    Parameters
    ----------
    epsilon : [type]
        [description]
    """
    path = "/Users/angelinabasova/Library/CloudStorage/OneDrive-bwedu/bomberman_rl_dev/agent_code/OWN/train.py"

    with open(path) as f:
        lines = f.readlines()

    epsilon_i = 0
    for i, line in enumerate(lines):
        if "self.epsilon" in line:
            epsilon_i = i
            break

    with open(path, "w") as f:
        eps = str(epsilon)
        lines[epsilon_i] = "    self.epsilon = " + eps + "\n"
        f.write("".join(lines))


def print_q_table_info(file: str):
    """Prints the number of states in the q table 

    Parameters
    ----------
    file : str
        [description]
    """
    path = "/Users/angelinabasova/Library/CloudStorage/OneDrive-bwedu/bomberman_rl_dev/agent_code/OWN/" + file
    path = path.replace("\"", "")
    with open(path, "rb") as file:
        q_table = pickle.load(file)
    print(f"Q-table has {len(q_table)} states")


if __name__ == "__main__":
    main()
