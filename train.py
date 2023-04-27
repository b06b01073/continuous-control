import gym
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt

from ddpg import Agent
import replay_buffer


writer = SummaryWriter()


def train(args):
    env = gym.make(args.env)
    action_dim, obs_dim = env.action_space.shape[0], env.observation_space.shape[0]

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim, action_low=env.action_space.low, action_high=env.action_space.high, noise=args.scale)
    buffer = replay_buffer.ReplayBuffer(capacity=args.capacity)

    total_rewards = []
    ma = []
    means = []
    check_pts_interval = 20

    for i in range(args.epoch):
        
        obs = env.reset()
        agent.reset_noise()


        # rollout and train
        agent.add_noise = True
        for c in range(args.cycles):
            for _ in range(args.rollout_steps):
                action = agent.step(obs)
                next_obs, reward, terminated, _ = env.step(action)

                buffer.append([obs, action, reward, next_obs, terminated])

                obs = next_obs

                if terminated:
                    obs = env.reset()
                    agent.reset_noise()
                    continue

            for _ in range(args.train_steps):
                agent.learn(buffer.sample(args.batch_size))
                agent.soft_update()

        # record episode result
        agent.add_noise = False
        obs = env.reset()
        total_reward = 0
        while True:
            action = agent.step(obs)
            next_obs, reward, terminated, _ = env.step(action)
            obs = next_obs
            total_reward += reward

            if terminated:
                break

        total_rewards.append(total_reward)
        ma.append(total_reward)
        if len(ma) > 30:
            ma = ma[1:]
        means.append(sum(ma) / len(ma))


        if i % check_pts_interval == 0:
            torch.save(agent.state_dict(), f'./params/{args.env}_{i}.pth')

    
        writer.add_scalar('Total Reward', total_reward, i)
        print(f'epoch {i}: {total_reward}')

    plt.plot(total_rewards, label='reward')
    plt.plot(means, label='moving average(last 30)')
    plt.title(f'Total reward({args.env})')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.savefig(f'result_{args.env}.jpg')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--capacity', '-c', type=int, default=100000)
    parser.add_argument('--scale', '-s', type=float, default=0.1)
    parser.add_argument('--rollout_steps', '-r', type=int, default=50)
    parser.add_argument('--train_steps', '-t', type=int, default=50)
    parser.add_argument('--cycles', type=int, default=20)

    args = parser.parse_args()

    train(args)