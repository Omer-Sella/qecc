"""
A module that contains the network architectures for the actor policy network and the value function.
Arguments needed:
num_cells
device


"""
from torch import nn


def create_actor_value_nets(num_cells, device):

    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Identity(),
        #nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        #nn.Tanh(),
        nn.Identity(),
        nn.LazyLinear(num_cells, device=device),
        #nn.Tanh(),
        nn.Identity(),
        nn.LazyLinear(env.action_spec.shape[-1], device=device),
    )
    return actor_net

def create_value_net(num_cells, device):
    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        #nn.Tanh(),
        nn.Identity(),
        nn.LazyLinear(num_cells, device=device),
        #nn.Tanh(),
        nn.Identity(),
        nn.LazyLinear(num_cells, device=device),
        #nn.Tanh(),
        nn.Identity(),
        nn.LazyLinear(1, device=device),
    )
    return value_net