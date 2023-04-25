import collections
import random
import torch

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, item):
        '''the items in the buffer is of the form ([obs, action, reward, next_obs, terminated])
        '''
        self.buffer.append(item)

    def sample(self, batch_size):
        # batch is of the form (batch_size, [obs, action, reward, next_obs, terminated])
        # each item in the batch is of the form ([obs, action, reward, next_obs, terminated])
        if batch_size > len(self.buffer):
            return None

        batch = random.sample(self.buffer, batch_size)

        obs = torch.stack([torch.from_numpy(experience[0]) for experience in batch]).to(torch.float32)
        actions = torch.stack([torch.from_numpy(experience[1]) for experience in batch]).to(torch.float32)
        rewards = torch.Tensor([experience[2] for experience in batch])
        next_obs = torch.stack([torch.from_numpy(experience[3]) for experience in batch]).to(torch.float32)
        terminated = torch.Tensor([experience[4] for experience in batch]).long()

        return obs, actions, rewards, next_obs, terminated

    
    def __len__(self):
        return len(self.buffer)
        