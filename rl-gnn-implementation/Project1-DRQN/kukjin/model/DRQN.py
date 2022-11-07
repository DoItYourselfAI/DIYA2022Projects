import torch
import torch.nn as nn
import torch.nn.functional as F
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DRQN(nn.Module):
    def __init__(self, C, H, W, K, S, num_actions):
        self.num_actions = num_actions
        super().__init__()
        def conv2d_size_out(size, kernel_size=K, stride=S):
          return (size - (kernel_size - 1) - 1) // stride + 1
      
      # 원래 브레이크아웃 환경은 state shape가 (3, 210, 160)이어서 가로 세로에 대해서 다르게 사이즈를 계산해야함
      # 전처리, 프레임스택을 통해 (4, 84, 84)로 바꿨으므로 너비, 높이가 동일해서 H 변수를 사용하지 않음
        convw = conv2d_size_out(W, K, S)
        convw = conv2d_size_out(convw, K, S)
        convw = conv2d_size_out(convw, K, S)
        
        self.channels = [C, 16, 32, 64]
        self.gru_i_dim = 64  # input dimension of GRU
        self.gru_h_dim = 64  # output dimension of GRU
        self.gru_N_layer = 1  # number of layers of GRU
        
        # Conv Layers
        self.layers = nn.ModuleList()
        for i in range(3):
          conv_layer = nn.Sequential(
          nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=K, stride=S, bias=True),
          nn.BatchNorm2d(self.channels[i+1]),
          nn.LeakyReLU())
          self.layers.append(conv_layer)
        self.conv_layers = nn.Sequential(*self.layers)
        
        # Conv2Gru layer
        linear_input_size = convw*convw*self.channels[-1]
        print(linear_input_size) 
        self.conv2gru = nn.Sequential(
            nn.Linear(linear_input_size, self.gru_i_dim),
            nn.LeakyReLU())
        
        # GRU Layers
        self.gru_layer = nn.GRU(input_size=self.gru_i_dim,
                          hidden_size=self.gru_h_dim,
                          num_layers=self.gru_N_layer,
                          batch_first='True')
        
        # FC Layer
        self.head = nn.Sequential(
            nn.Linear(self.gru_h_dim, self.num_actions),
            nn.LeakyReLU(),
        )

    def forward(self, x, hidden):
      if len(x.shape) < 4:
          x = x.unsqueeze(0)
      x = self.conv_layers(x)
      x = x.view(x.size(0), -1)
      x = self.conv2gru(x)
      if len(x.shape) < 3:
        x = x.unsqueeze(0) # 1, 1, 64
    # ! FIX HERE 
      x, new_hidden = self.gru_layer(x, hidden)
      out = self.head(x)
      return out, new_hidden
    
    def sample_action(self, x, hidden, eps):
        out, new_hidden = self.forward(x, hidden)
        coin = random.random()
        if coin < eps:
            # ! DRQN에서 엡실론 그리디를 적용하면 히든도 반환해야하는가?
            return random.randint(0, self.num_actions-1), new_hidden
        else:
            return torch.argmax(out).item(), new_hidden
    
    def init_hidden_state(self, batch_size, training=None):
        if training is True:
            return torch.zeros([batch_size, 1, self.gru_h_dim], device=device)
        else:
            return torch.zeros([1, 1, self.gru_h_dim], device=device)

        

def train_drqn(behavior_net, target_net, memory, optimizer, gamma, batch_size):
    state, action, reward, next_state, done = memory.sample(batch_size)
    state = state.to(device)
    action = action.to(device)
    reward = reward.to(device)
    next_state = next_state.to(device)
    done = done.to(device)
    done = done.to(int)
    
    q_out = behavior_net(state)
    q_a = q_out.gather(1, action)
    # action 축 기준으로 max 취한 후, 0번째 인덱스를 가져오면 values를 가져온다.
    # 그 후 shape를 맞추기 위해 unsqueeze()를 한다.
    max_target_q = target_net(next_state).max(1)[0].unsqueeze(1)
    target = reward + gamma * max_target_q * (torch.ones_like(done) - done)
    loss = F.smooth_l1_loss(q_a, target).to(device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
