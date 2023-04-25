import gym
from argparse import ArgumentParser
import torch

from ddpg import Agent

def train(args):
    env = gym.make(args.env, render_mode='human')
    action_dim, obs_dim = env.action_space.shape[0], env.observation_space.shape[0]

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim, action_low=env.action_space.low, action_high=env.action_space.high, add_noise=False)
    agent.load_state_dict(torch.load(args.params))

    while True:
        obs = env.reset()
        while True:
            action = agent.step(obs)
            next_obs, reward, terminated, _ = env.step(action)
            obs = next_obs
            if terminated:
                break

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='LunarLanderContinuous-v2') # HalfCheetah-v4
    parser.add_argument('--params', '-p')

    args = parser.parse_args()

    train(args)