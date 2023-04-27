import numpy as np 

class GaussianNoise():
    def __init__(self, size, scale,tau=0.01):
        self.noise = None
        self.size = size
        self.tau = tau
        self.scale = scale

    def reset(self):
        self.noise = np.random.normal(size=self.size, scale=self.scale)

    def sample(self):
        self.noise = np.random.normal(size=self.size, scale=self.scale) * self.tau + self.noise * (1 - self.tau)
        return self.noise
