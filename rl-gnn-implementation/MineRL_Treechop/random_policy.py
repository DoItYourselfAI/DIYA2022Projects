import gym

# noinspection PyUnresolvedReferences
import minerl

from utils import random_policy


def main():
    env = gym.make("MineRLTreechop-v0")
    env.make_interactive(port=6666, realtime=True)
    print(env.observation_space)
    print(env.action_space)
    for e in range(10):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = random_policy(env)
            next_obs, reward, done, info = env.step(action)
            score += reward

            if done:
                print(f"Episode {e} is finished")
                print(f"Total score: {score}")
                break


if __name__ == "__main__":
    main()
