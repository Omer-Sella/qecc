"""
A module that contains the network architectures for the actor policy network and the value function.
Arguments needed:
num_cells
device


"""
from torch import nn


def create_actor_value_nets(action_spec, num_cells, device = None):

    if device is not None:
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
            nn.LazyLinear(action_spec.shape[-1], device=device),
        )
    else:
        actor_net = nn.Sequential(
            nn.LazyLinear(num_cells),
            nn.Identity(),
            #nn.Tanh(),
            nn.LazyLinear(num_cells),
            #nn.Tanh(),
            nn.Identity(),
            nn.LazyLinear(num_cells),
            #nn.Tanh(),
            nn.Identity(),
            nn.LazyLinear(action_spec.shape[-1]),
        )

    return actor_net

def create_value_net(num_cells, device = None):
    if device is not None:
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
    else:
            value_net = nn.Sequential(
            nn.LazyLinear(num_cells),
            #nn.Tanh(),
            nn.Identity(),
            nn.LazyLinear(num_cells),
            #nn.Tanh(),
            nn.Identity(),
            nn.LazyLinear(num_cells),
            #nn.Tanh(),
            nn.Identity(),
            nn.LazyLinear(1),
        )
    return value_net