import logging
import os
import random

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger()


def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def save_model(episode, SAVE_PERIOD, SAVE_PATH, model, MODEL_NAME, ENV_NAME):
    if episode % SAVE_PERIOD == 0:
        save_path_name = (
            SAVE_PATH + ENV_NAME + "_" + MODEL_NAME + "_" + str(episode) + ".pt"
        )
        torch.save(model.state_dict(), save_path_name)
        logger.info("model saved")


def load_model(model, SAVE_PATH, MODEL_NAME):
    model.load_state_dict(torch.load(SAVE_PATH + MODEL_NAME + ".pt"))
    logger.info("load model successfully")
    return model


def converter(observation, device):
    # Convert pixels
    pixels = observation["pov"]
    pixels = torch.from_numpy(pixels).float()  # 64, 64, 3
    pixels /= 255.0  # int2float
    pixels = pixels.permute(2, 0, 1)  # 3, 64, 64
    if len(pixels.shape) < 4:  # Add batch dimension to pixels
        pixels = pixels.unsqueeze(0)  # 1, 3, 64, 64

    return pixels.to(device, dtype=torch.float)


def random_policy(env):
    action_index = random.randint(0, 8)
    action = make_8action(env, action_index)
    action = env.action_space.sample()
    return action


def make_8action(env, action_index):
    # Action들을 정의
    action = env.action_space.noop()

    # No action
    if action_index == 0:
        action["camera"] = [0, 0]
        action["forward"] = 0
        action["jump"] = 0

    # Camera
    # action['camera']  DQN 0 노드 0, 1, 2, 3, 4
    # action['forward'] 1 노드 0, 1

    elif action_index == 1:
        action["camera"] = [0, -10]  # y츅
    elif action_index == 2:
        action["camera"] = [0, 10]
    elif action_index == 3:
        action["camera"] = [-10, 0]  # x축
    elif action_index == 4:
        action["camera"] = [10, 0]

    elif action_index == 5:
        action["attack"] = 1

    # Move forward or jump
    elif action_index == 6:
        action["forward"] = 1
    elif action_index == 7:
        action["jump"] = 1

    return action
