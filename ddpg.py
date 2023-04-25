import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np


from random_process import OrnsteinUhlenbeckProcess

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'training on {device}')

class Agent:
    def __init__(self, obs_dim, action_dim, action_low, action_high, gamma=0.99, tau=0.01):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high


        self.actor = Actor(input_dim=self.obs_dim, output_dim=self.action_dim, scale=np.max(action_high)).to(device)
        self.critic = Critic(input_dim=self.obs_dim + self.action_dim, output_dim=1).to(device)

        self.target_actor = Actor(input_dim=self.obs_dim, output_dim=self.action_dim, scale=np.max(action_high)).to(device)
        self.target_critic = Critic(input_dim=self.obs_dim + self.action_dim, output_dim=1).to(device)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.ou_process = OrnsteinUhlenbeckProcess(size=self.action_dim)

        self.mse = nn.MSELoss()

        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=1e-2, weight_decay=1e-2)
        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=1e-3)


    def step(self, obs):
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(torch.float32).to(device)
            action = self.actor(obs).cpu().detach().numpy()
            noise = self.ou_process.sample()

            return np.clip(action, a_max=self.action_high, a_min=self.action_low)
        
    def learn(self, experiences):
        if experiences is None:
            return
        
        obs, actions, rewards, next_obs, terminated = experiences
        obs, actions, rewards, next_obs, terminated = obs.to(device), actions.to(device), rewards.to(device), next_obs.to(device), terminated.to(device)
        

        self.update_actor(obs)
        self.update_critic(obs, actions, rewards, next_obs, terminated)

    def update_critic(self, obs, actions, rewards, next_obs, terminated):
        y = rewards + (1 - terminated) * self.gamma * self.target_critic(next_obs, self.target_actor(next_obs)).squeeze().detach()
        q = self.critic(obs, actions).squeeze()

        self.critic_optim.zero_grad()
        loss = self.mse(y, q)
        loss.backward()

        nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optim.step()



    def update_actor(self, obs):
        grad = -self.critic(obs, self.actor(obs)).mean() # negative sign for gradient ascend
        
        self.actor_optim.zero_grad()
        grad.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optim.step()

        
    def soft_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim0 = 400
        self.hidden_dim1 = 200
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=self.hidden_dim0),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim0, out_features=self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim1, out_features=output_dim),
        )

    def forward(self, obs, action):
        input = torch.cat((obs, action), dim=1)
        return self.net(input)
    
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128, scale=1):
        super().__init__()
        self.scale = scale
        self.hidden_dim0 = 400
        self.hidden_dim1 = 200
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=self.hidden_dim0),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim0, out_features=self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim1, out_features=output_dim),
        )

    def forward(self, obs):
        obs = self.scale * nn.Tanh()(self.net(obs))
        return obs
