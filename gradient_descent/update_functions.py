from gradient_descent.interfaces import UpdateFunction
from model.system_model import SystemModel
import numpy as np
import math


class GradientNoise(UpdateFunction):
    def __init__(self, lr, gamma=1.2, seed=1):
        self.lr = lr
        self.gamma = gamma
        self.seed = seed
        self.rng = None
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def update(self, S: SystemModel, x: [float], nabla: [float], t: int) -> [float]:
        # noise added to the gradients helps with the optimization
        # the noise decays with the iterations
        # for big systems (e.g. 10x10x5), it is beneficial to reduce the noise added, so
        # I added a reducing factor to the noise (len(nabla)): bigger systems need less noise
        # for smaller systems, this reduction seems to not affect negatively
        std = self.lr / (1 + t + len(nabla)) ** self.gamma
        noise = self.rng.normal(0, std, len(nabla))
        for j in range(len(nabla)):
            nabla[j] += noise[j]
        return nabla


class Adam(UpdateFunction):
    def __init__(self, lr=3, beta1=0.9, beta2=0.999, epsilon=0.1):
        self.size = None
        self.m = None
        self.v = None
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.size = None
        self.m = None
        self.v = None

    def update(self, S: SystemModel, x: [float], nabla: [float], t: int) -> [float]:
        if not self.size:
            self.size = len(nabla)
            self.m = [0]*self.size
            self.v = [0]*self.size

        updates = [0]*self.size
        for i in range(self.size):
            self.m[i] = self.beta1 * self.m[i] + (1 + self.beta1) * nabla[i]
            self.v[i] = self.beta2 * self.v[i] + (1 + self.beta2) * nabla[i] ** 2

            me = self.m[i] / (1 - self.beta1 ** t)
            ve = self.v[i] / (1 - self.beta2 ** t)

            updates[i] = -self.lr*me/(math.sqrt(ve)+self.epsilon)

        return updates


class NoisyAdam(UpdateFunction):
    def __init__(self, lr=3, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=0.9, seed=1):
        self.noise = GradientNoise(lr=lr, gamma=gamma, seed=seed)
        self.adam = Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)

    def reset(self):
        self.noise.reset()
        self.adam.reset()

    def update(self, S: SystemModel, x: [float], nabla: [float], t: int) -> [float]:
        noisy_gradient = self.noise.update(S, x, nabla, t)
        update = self.adam.update(S, x, noisy_gradient, t)
        return update
