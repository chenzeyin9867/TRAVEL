import numpy as np
import torch


class RunningMeanStd(object):
    def __init__(self, shape, epsilon = 1e-4):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon
        self.epsilon = 1e-8
    def update(self, arr) -> None:
        batch_mean = torch.mean(torch.squeeze(arr, dim=1), axis=0)
        batch_var = torch.var(torch.squeeze(arr, dim=1), axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def process(self, obs):
        return ((obs - self.mean) / np.sqrt(self.var + 1e-8)).to(torch.float32)
        

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
