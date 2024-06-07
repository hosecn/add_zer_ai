import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 超参数
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 500
LR = 0.001
BATCH_SIZE = 32
MEMORY_CAPACITY = 10000

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN:
    def __init__(self, state_dim, action_dim):
        self.eval_net = QNet(state_dim, action_dim)
        self.target_net = QNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.memory = []
        self.pointer = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(np.array(state)).unsqueeze(0)
            action_values = self.eval_net(state)
            return torch.argmax(action_values).item()

    def store_transition(self, state, action, reward, next_state):
        state = np.array(state)
        next_state = np.array(next_state)
        if len(self.memory) < MEMORY_CAPACITY:
            self.memory.append(None)
        self.memory[self.pointer] = (state, action, reward, next_state)
        self.pointer = (self.pointer + 1) % MEMORY_CAPACITY

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))

        q_values = self.eval_net(states).gather(1, actions).squeeze(1)
        max_next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + GAMMA * max_next_q_values

        loss = self.loss_func(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(EPSILON_END, self.epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY)

    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
