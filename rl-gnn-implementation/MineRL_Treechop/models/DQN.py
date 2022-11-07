import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, num_actions=8):
        self.num_actions = num_actions
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.bn4 = nn.BatchNorm2d(128)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(64, 8, 4)
        convw = conv2d_size_out(convw, 4, 2)
        convw = conv2d_size_out(convw, 3, 1)
        convw = conv2d_size_out(convw, 2, 1)

        linear_input_size = convw * convw * 128
        self.linear = nn.Linear(linear_input_size, 1024)
        self.head = nn.Linear(1024, self.num_actions)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0).to(device=device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        x = self.head(x)  # view는 numpy의 reshape 와 같다.
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)

        coin = np.random.rand(1)
        if coin < epsilon:
            return np.random.randint(0, self.num_actions - 1)
        else:
            return torch.argmax(out)


# TODO: Implement Update Q network function


def update_q_network(buffer, batch_size, behavior_net, target_net, optimizer):
    batch, weights, tree_idxs = buffer.sample(batch_size)
    print(batch)
    print(weights)
    state, action, reward, next_state, done = batch
    print(state.shape)
    print(action.shape)
    print(reward.shape)

    # ? shape information
    # ? state: [batch, 3, 64, 64]
    # ? action: [batch, 1]
    # ? reward: [batch, 1]
    # ? next_state: [batch, 3, 64, 64]
    # ? done: [batch, 1]

    q_value = behavior_net


def update(batch, weights=None):
    state, action, reward, next_state, done = batch

    Q_next = self.target_model(next_state).max(dim=1).values
    Q_target = reward + self.gamma * (1 - done) * Q_next
    Q = self.model(state)[torch.arange(len(action)), action.to(torch.long).flatten()]

    assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

    if weights is None:
        weights = torch.ones_like(Q)

    td_error = torch.abs(Q - Q_target).detach()
    loss = torch.mean((Q - Q_target) ** 2 * weights)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    with torch.no_grad():
        self.soft_update(self.target_model, self.model)

    return loss.item(), td_error


def train_network(memory, alpha=0.6, beta=0.4, batch_size=32, run_step=1e6):
    transitions, weights, indices, sampled_p, mean_p = memory.sample(beta, batch_size)

    for key in transitions.keys():
        transitions[key] = self.as_tensor(transitions[key])

    state = transitions["state"]
    action = transitions["action"]
    reward = transitions["reward"]
    next_state = transitions["next_state"]
    done = transitions["done"]

    eye = torch.eye(self.action_size).to(self.device)
    one_hot_action = eye[action.view(-1).long()]
    q = (self.network(state) * one_hot_action).sum(1, keepdims=True)

    with torch.no_grad():
        max_Q = torch.max(q).item()
        next_q = self.network(next_state)
        max_a = torch.argmax(next_q, axis=1)
        max_eye = torch.eye(self.action_size).to(self.device)
        max_one_hot_action = eye[max_a.view(-1).long()]

        next_target_q = self.target_network(next_state)
        target_q = reward + (next_target_q * max_one_hot_action).sum(
            1, keepdims=True
        ) * (self.gamma * (1 - done))

    # Update sum tree
    td_error = abs(target_q - q)
    p_j = torch.pow(td_error, self.alpha)
    for i, p in zip(indices, p_j):
        self.memory.update_priority(p.item(), i)

    weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)

    loss = (weights * (td_error ** 2)).mean()
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    self.optimizer.step()

    self.num_learn += 1

    result = {
        "loss": loss.item(),
        "epsilon": self.epsilon,
        "beta": self.beta,
        "max_Q": max_Q,
        "sampled_p": sampled_p,
        "mean_p": mean_p,
    }
    return result


def soft_update(target_net, behavior_net):
    for tp, sp in zip(target_net.parameters(), behavior_net.parameters()):
        tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)


def hard_update(behavior_net, target_net):
    target_net.load_state_dict(behavior_net.state_dict())
