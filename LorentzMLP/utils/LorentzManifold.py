import torch
from geoopt.manifolds.lorentz import Lorentz


class LorentzManifold(Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(LorentzManifold, self).__init__(k=k, learnable=learnable)

    def add_time(self, x_space):
        """ Adds time component to given space components of vector x, and produces lorentzian vector x containing both time and space dimension. """
        x_time = self.calc_time(x_space)
        return torch.cat([x_time, x_space], dim=-1)

    def calc_time(self, x_space):
        """ Calculates time component from given space component. """
        return torch.sqrt((x_space ** 2).sum(dim=-1, keepdim=True) + self.k)