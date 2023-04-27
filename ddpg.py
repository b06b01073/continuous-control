import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np

from random_process import OrnsteinUhlenbeckProcess

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'training on {device}')


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, action_low, action_high, gamma=0.99, tau=0.001, add_noise=True):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high

        self.actor = Actor(input_dim=self.obs_dim, output_dim=self.action_dim, scale=np.max(action_high)).to(device)
        self.critic = Critic(obs_dim=self.obs_dim, action_dim=self.action_dim, output_dim=1).to(device)

        self.target_actor = Actor(input_dim=self.obs_dim, output_dim=self.action_dim, scale=np.max(action_high)).to(device)
        self.target_critic = Critic(obs_dim=self.obs_dim, action_dim=self.action_dim, output_dim=1).to(device)

        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        self.random_process = OrnsteinUhlenbeckProcess(size=action_dim, theta=0.15, mu=0.0, sigma=0.1)

        self.mse = nn.MSELoss()

        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=1e-3, weight_decay=1e-2)
        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=1e-4)


        self.add_noise = add_noise

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


    def step(self, obs):
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(torch.float32).to(device)
            action = self.actor(obs).cpu().detach().numpy()

            if self.add_noise:
                action += self.random_process.sample()

            return np.clip(action, a_max=self.action_high, a_min=self.action_low)
        
    def reset_noise(self):
        self.random_process.reset_states()
        
    def learn(self, experiences):
        if experiences is None:
            return
        
        obs, actions, rewards, next_obs, terminated = experiences
        obs, actions, rewards, next_obs, terminated = obs.to(device), actions.to(device), rewards.to(device), next_obs.to(device), terminated.to(device)

        self.update_actor(obs)
        self.update_critic(obs, actions, rewards, next_obs, terminated)

    def update_critic(self, obs, actions, rewards, next_obs, terminated):
        y = rewards + (1 - terminated) * self.gamma * self.target_critic(next_obs, self.target_actor(next_obs)).squeeze()
        q = self.critic(obs, actions).squeeze()

        self.critic_optim.zero_grad()
        loss = self.mse(y.detach(), q)
        loss.backward()

        self.critic_optim.step()



    def update_actor(self, obs):
        self.actor_optim.zero_grad()
        grad = -self.critic(obs, self.actor(obs)).mean() # negative sign for gradient ascend
        grad.backward()
        self.actor_optim.step()

        
    def soft_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class Critic(nn.Module):
    def __init__(self, action_dim, obs_dim, output_dim):
        super().__init__()
        self.hidden_dim0 = 400
        self.hidden_dim1 = 300
        self.fc1 = nn.Linear(in_features=obs_dim, out_features=self.hidden_dim0)
        self.fc2 = nn.Linear(in_features=self.hidden_dim0 + action_dim, out_features=self.hidden_dim1)
        self.fc3 = nn.Linear(in_features=self.hidden_dim1, out_features=output_dim)

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, action):
        output = self.relu(self.fc1(obs))
        output = torch.cat((output, action), dim=1)
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return output
    
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim=1, scale=1):
        super().__init__()
        self.scale = scale
        self.hidden_dim0 = 400
        self.hidden_dim1 = 300
        self.fc1 = nn.Linear(in_features=input_dim, out_features=self.hidden_dim0)
        self.fc2 = nn.Linear(in_features=self.hidden_dim0, out_features=self.hidden_dim1)
        self.fc3 = nn.Linear(in_features=self.hidden_dim1, out_features=output_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        output = self.relu(self.fc1(obs))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return self.tanh(output) * self.scale


