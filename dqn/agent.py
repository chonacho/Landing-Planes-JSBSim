import jsbgym
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.tau = 0.001

        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return np.random.uniform(-1, 1, self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.cpu().data.numpy()[0]
            c = np.clip(action,-1,1)
            if not np.array_equal(c, [-1, 1, 1]): print(c)
            return np.clip(action, -1, 1)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.vstack([x[0] for x in minibatch])).to(self.device)
        actions = torch.FloatTensor(np.vstack([x[1] for x in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.vstack([x[2] for x in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.vstack([x[3] for x in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.vstack([x[4] for x in minibatch])).to(self.device)

        current_q_values = self.policy_net(states)
        next_q_values = self.target_net(next_states)

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, episode):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.checkpoint_dir}/dqn_checkpoint_ep{episode}_{timestamp}.pth"
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode
        }, filename)
        return filename

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        return checkpoint['episode']

def train():
    env = gym.make("C172-HeadingControlTask-Shaping.STANDARD-NoFG-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = DQNAgent(state_size, action_size)

    episodes = 1000000
    checkpoint_frequency = 10000
    print_frequency = 100

    rewards_since_print = []
    losses_since_print = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_rewards = []

        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            episode_rewards.append(reward)
            state = next_state

            if done:
                break

        loss = agent.replay()

        episode_avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        rewards_since_print.append(episode_avg_reward)
        if loss is not None:
            losses_since_print.append(loss)

        if (episode + 1) % print_frequency == 0:
            avg_reward = sum(rewards_since_print) / len(rewards_since_print)
            avg_loss = sum(losses_since_print) / len(losses_since_print) if losses_since_print else 0
            print(f"Episode: {episode + 1}, Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
            rewards_since_print = []
            losses_since_print = []

        if (episode + 1) % checkpoint_frequency == 0:
            checkpoint_path = agent.save(episode + 1)
            print(f"Saved checkpoint to {checkpoint_path}")

    env.close()

if __name__ == "__main__":
    train()
