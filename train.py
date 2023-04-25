import gym
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from ddpg import Agent
import replay_buffer


writer = SummaryWriter()

def train(args):
    env = gym.make('LunarLanderContinuous-v2', continuous=True, render_mode='human')
    action_dim, obs_dim = env.action_space.shape[0], env.observation_space.shape[0]

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim, action_low=env.action_space.low, action_high=env.action_space.high)
    buffer = replay_buffer.ReplayBuffer(capacity=20000)


    for i in range(args.epoch):
        total_reward = 0
        obs = env.reset()
        while True:
            action = agent.step(obs)
            next_obs, reward, terminated, _ = env.step(action)


            buffer.append([obs, action, reward, next_obs, terminated])

            obs = next_obs
            agent.learn(buffer.sample(args.batch_size))
            agent.soft_update()

            total_reward += reward

            if terminated:
                break
        writer.add_scalar('Total Reward', total_reward, i)
        print(f'episode {i}: {total_reward}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--epoch', '-e', type=int, default=200)

    args = parser.parse_args()

    train(args)