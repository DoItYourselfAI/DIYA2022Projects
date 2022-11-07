import gym
import numpy as np
import random
import torch
import torch.optim as optim
from wrapper.framestack import FrameBuffer
from wrapper.preprocess import PreprocessAtari
from model.DRQN import DRQN, train_drqn
from buffer.replay_buffer import ReplayBuffer

# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# from scipy.stats import ks_2samp
# from IPython.display import Images



def make_env():
    env = gym.make("BreakoutDeterministic-v4")
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env

def random_policy(state):
    action = random.randint(0, 3)
    return action

def main():
    
    GAMMA=0.9
    EPISODES = 15000
    MEMORY_SIZE = 30000
    BATCH_SIZE = 32   # 32
    LEARNING_RATE = 0.001   # 0.01
    TARGET_UPDATE = 10  # 5

    startEpsilon = 1.0
    endEpsilon = 0.05
    total_episodes = 100
    epsilon = startEpsilon
    stepDrop = (startEpsilon - endEpsilon) / total_episodes


    env = make_env()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape
    print(n_actions) # 4
    print(state_dim) # (4, 84, 84)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    behavior_policy = DRQN(4, 84, 84, 3, 2, 4).to(device)     # C, H, W, K, S, num_actions
    target_policy = DRQN(4, 84, 84, 3, 2, 4).to(device) 
    target_policy.load_state_dict(behavior_policy.state_dict())
    optimizer = optim.Adam(behavior_policy.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)

    for episode in range(total_episodes):
        hidden = behavior_policy.init_hidden_state(BATCH_SIZE, False)
        if(epsilon > endEpsilon):
            epsilon -= stepDrop
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            action, new_hidden = behavior_policy.sample_action(torch.tensor(state).to(device), hidden, epsilon)
            next_state, reward, done, info = env.step(action)
            
            transition = (state, action, reward, next_state, done)
            memory.put(transition)
            state = next_state
            hidden = new_hidden
            total_reward += reward
            
            if done:
                print(f'Total rewards of episode {episode}: {total_reward}')
                print(f"# of transitions in memory: {memory.size()}")
                break
            
            #if memory.size() > 2000:
            #    train_drqn(behavior_policy, target_policy, memory, optimizer,GAMMA, BATCH_SIZE)
        if episode % TARGET_UPDATE == 0:
            target_policy.load_state_dict(behavior_policy.state_dict())
                
main()
    