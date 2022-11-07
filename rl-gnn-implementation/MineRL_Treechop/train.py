import logging

import gym

# noinspection PyUnresolvedReferences
import minerl
import torch
import torch.optim as optim

from buffers.buffer import PrioritizedReplayBuffer
from models.DQN import DQN, update_q_network
from utils import make_8action, converter, set_seed

logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(device)


def main():
    total_episodes = 100
    start_epsilon = 0.95
    end_epsilon = 0.05
    epsilon = start_epsilon
    step_drop = (start_epsilon - end_epsilon) / total_episodes

    env = gym.make("MineRLTreechop-v0")
    env.make_interactive(port=6666, realtime=True)
    set_seed(env, seed=3154)
    behavior_net = DQN().to(device, dtype=torch.float)
    target_net = DQN().to(device, dtype=torch.float)
    lr = 0.0005
    memory = PrioritizedReplayBuffer(
        state_size=(3, 64, 64),
        action_size=1,
        buffer_size=20000,
        eps=1e-2,
        alpha=0.1,
        beta=0.1,
    )
    optimizer = optim.Adam(behavior_net.parameters(), lr)
    count = 0

    for e in range(total_episodes):
        obs = env.reset()
        done = False
        score = 0
        if epsilon > end_epsilon:
            epsilon -= step_drop

        while not done:
            count += 1
            env.render()
            action_index = behavior_net.sample_action(converter(obs, device), epsilon)
            action = make_8action(env, action_index)
            next_obs, reward, done, info = env.step(action)
            score += reward
            transition = (
                converter(obs, device),
                action_index,
                reward,
                converter(next_obs, device),
                done,
            )

            # TODO: save transition in PER
            memory.add(transition)
            logger.info(memory.size)

            obs = next_obs
            if done:
                logger.info(f"Episode {e} is finished")
                logger.info(f"Total score: {score}")
                break

            # TODO: Update Q network
            if count > memory.count:
                update_q_network(memory, 32, behavior_net, target_net, optimizer)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
