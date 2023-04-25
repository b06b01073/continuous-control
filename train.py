import gym
import numpy as np
from argparse import ArgumentParser
import torch

from ddpg import Agent
import replay_buffer

def train(args):
    env = gym.make('Pendulum-v1')
    action_dim, obs_dim = env.action_space.shape[0], env.observation_space.shape[0]
    agent = Agent(obs_dim=obs_dim, action_dim=action_dim)
    buffer = replay_buffer.ReplayBuffer(capacity=10000)


    for i in range(args.epoch):
        total_reward = 0
        obs = env.reset()
        while True:
            action = agent.step(obs)
            next_obs, reward, terminated, _ = env.step(action)

            buffer.append([obs, action, reward, next_obs, terminated])

            obs = next_obs
            agent.learn(buffer.sample(args.batch_size))

            total_reward += reward

            if terminated:
                env.close()
                break
        print(f'epoch {i}: {total_reward}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--epoch', '-e', type=int, default=1000)

    args = parser.parse_args()

    train(args)