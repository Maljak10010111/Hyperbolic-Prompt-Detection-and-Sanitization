import torch
from code.lib.geoopt import Lorentz


class LorentzManifold(Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(LorentzManifold, self).__init__(k=k, learnable=learnable)

    def add_time(self, space):
        """ Adds time component to given space components and produces lorentzian model vector """
        time = self.calc_time(space)
        return torch.cat([time, space], dim=-1)

    def calc_time(self, space):
        """ Calculates time component from given space component. """
        return torch.sqrt(torch.norm(space, dim=-1, keepdim=True)**2+self.k)
        # return torch.sqrt(torch.clamp(torch.norm(space, dim=-1, keepdim=True) ** 2 + self.k, min=1e-5))
        # square_norm = (space ** 2).sum(dim=-1, keepdim=True)  # No sqrt involved
        # safe_val = torch.clamp(square_norm + self.k, min=1e-5)
        # return torch.sqrt(safe_val)
