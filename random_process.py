import numpy as np 

class GaussianNoise():
    def __init__(self, size, tau=0.01):
        self.noise = None
        self.size = size
        self.tau = tau

    def reset(self):
        self.noise = np.random.normal(size=self.size, scale=0.1)

    def sample(self):
        self.noise = np.random.normal(size=self.size, scale=0.1) * self.tau + self.noise * (1 - self.tau)
        return self.noise
