"""
A module that contains the network architectures for the actor policy network and the value function.
Arguments needed:
num_cells
device


"""
from torch import nn
import torch
from qecc.attentionArchitectures import CodeEncoder, AttentionPool, buildTokenFeatures
import numpy as np

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

############ new model architecture, with tanh and pretrained encoder, no lazy initialization.
############ In the new model we need to remove the normalization (so no transforming the environment to -1,1 outputs).

class hybridNet(nn.Module):
    """Two-branch net for the hybrid observation [aX|aY|bX|bY | A|B-flattened].

    exponent bits (first 2l+2m, raw 0/1) -> pretrained attention encoder -> (dModel)
    A|B block (rest, scaled to +-1)      -> MLP trunk (your current net)  -> (num_cells)
    concat -> fusion -> outputSize  (actor: 2l+2m+4 logits; critic: 1)
    """

    def __init__(self, l, m, 
                 outputSize, 
                 minimumNumberOfQubits,
                 surrogateModelPath, 
                 num_cells=256,
                 device=None):
        super().__init__()
        if surrogateModelPath is None:
            raise ValueError("Surrogate model is required for this architecture. No path to model weights given.")
        else:
            checkpoint = torch.load(surrogateModelPath, map_location="cpu", weights_only=False)
                
        self.minimumNumberOfQubits = minimumNumberOfQubits
        self.numberOfFeaturesForLogicalQubits = 3   # 3 was a design choice: deficit, regime indicator, log1p(k). This matches kFeatures in forward

        self.dModel = checkpoint["hyperParameters"]["dModel"]
        self.nHead =  checkpoint["hyperParameters"]["nHead"]
        self.numLayers= checkpoint["hyperParameters"]["numLayers"]
        self.dimFeedforward = checkpoint["hyperParameters"]["dimFeedforward"]
        self.numberOfHarmonics=checkpoint["hyperParameters"]["numberOfHarmonics"]
        featureSizes = [1, # bit value — 0 or 1: is this coefficient set in the polynomial?
                        4, # group one-hot — [1,0,0,0]=aX, [0,1,0,0]=aY, [0,0,1,0]=bX, [0,0,0,1]=bY
                        2 * self.numberOfHarmonics, # cyclic position — the pair (sin, cos) at each harmonic h=1,2,3 of the slot angle θ = 2πi/p: [sin θ, cos θ, sin 2θ, cos 2θ, sin 3θ, cos 3θ] where period p = l for aX/bX, p = m for aY/bY
                        1, # linear position — i/p ∈ [0, 1)
                        3] # globals (identical on every token) — [log l, log m, 1.0]
        self.featureSize =  np.sum(featureSizes)
        self._l = l
        self._m = m
        
        self.numberOfBitsForPolynomialExponents = 2 * l + 2 * m
        self.codeSize = 2 * (l * m) ** 2


        self.encoder = CodeEncoder(featureSize=self.featureSize, dModel=self.dModel, nHead = self.nHead,
                                   numLayers=self.numLayers, dimFeedforward=self.dimFeedforward)
        self.pool = AttentionPool(self.dModel)
        # Now we actually need to load the pretrained model
        stateOfTrainedSurrogate = checkpoint["state_dict"]
        self.encoder.load_state_dict(
            {k[len("encoder."):]: v for k, v in stateOfTrainedSurrogate.items() if k.startswith("encoder.")},
            strict=True)
        self.pool.load_state_dict(
            {k[len("pool."):]: v for k, v in stateOfTrainedSurrogate.items() if k.startswith("pool.")},
            strict=True)
        
        # This is similar to the old MLP model, with lazy linear replaced with linear and with Tanh instead of identity.
        self.mlpBranch = nn.Sequential(
            nn.Linear(in_features=self.codeSize, out_features=num_cells), nn.Tanh(),
            nn.Linear(in_features=num_cells, out_features=num_cells), nn.Tanh(),
            nn.Linear(in_features=num_cells, out_features=num_cells), nn.Tanh(),
        )
        # The two branches converge here; 
        self.fusionLayer = nn.Sequential(
            nn.Linear(in_features=self.dModel + num_cells + self.numberOfFeaturesForLogicalQubits, out_features=num_cells), nn.Tanh(),
            nn.Linear(in_features=num_cells, out_features=outputSize),
        )
        if device is not None:
            self.to(device)


    def forward(self, aX, bX, aY, bY, code, numberOfLogicalQubits): 
        
        """
        forward pushes the observation components through the model, 
        in a parameter specific way.
        It also needs to handle different kinds of calls in different places in the training and evaluation. For example:
        Caller	                                            aX arrives as	k arrives as	Why
        policy_module(evaluationEnv.reset())	            (6,)	        (1,)	        single env, no batch dimension (this is the mode I usually have in my head)
        collection via ParallelEnv(num_workers = 48, …)	    (48, 6)	        (48, 1)	        one row per worker
        eval rollout of length 30	                        (30, 6)	        (30, 1)	        one row per time step
        PPO loss on a sub-batch of 64	                    (64, 6)	        (64, 1)	        one row per stored frame
        
        But that's not all ! It could be we have both a time axis, and a num_workers axis, so something like: (48,30,6) and (48,30,1) !!
        So generally the dimension will be (B,T,6) B== batch_size T == maxmimum_number_of_time_steps (or just B dimension, or just T, Or B,T,S, ... ,6)
        So consistently, the last dimension (-1), is the dimension I always have in my head,
        and consistently shape[:-1] (i.e.: all the numbers in .shape EXCEPT the last one !) will give all the other dimensions
        We can choose one of these (say aX) and get these dimension from aX.shape[:-1] !!!!

        """
        leadingShape = aX.shape[:-1] # Omer: So these are all the dimensions you don't want to think about
        aX = aX.reshape(-1, aX.shape[-1]) 
        aY = aY.reshape(-1, aY.shape[-1])
        bX = bX.reshape(-1, bX.shape[-1])
        bY = bY.reshape(-1, bY.shape[-1])
        code = code.reshape(-1, code.shape[-1])
        numberOfLogicalQubits = numberOfLogicalQubits.reshape(-1, 1)

        exponents = torch.cat([aX,aY,bX,bY],dim = -1) # Note the ordering ! This is compatible with the token builder, not the action !
        kFeatures = torch.cat([                       # computed from raw k, config-invariant
            (numberOfLogicalQubits - self.minimumNumberOfQubits).clamp(max=0.0) / self.minimumNumberOfQubits,   # deficit  (the penalty branch's shape)
            (numberOfLogicalQubits >= self.minimumNumberOfQubits).float(),                     # regime indicator
            torch.log1p(numberOfLogicalQubits),                               # scale-friendly magnitude
            ], dim=-1)

#        # Support any leading batch shape ((2616,), (B, 2616), (T, B, 2616)),
#        leadingShape = observation.shape[:-1]
#        flat = observation.reshape(-1, observation.shape[-1])

#        exponents = flat[:, :self.numberOfBitsForPolynomialExponents]            # raw 0/1 - this is what the encoder was pretrained on
        abBlock = code * 2.0 - 1.0  # This normalizes the 0/1 bits of the flat code to +-1, replacing the need to transform the environemnt using Transform.

        tokens = buildTokenFeatures(exponents, self._l, self._m, self.numberOfHarmonics)
        encoded = self.pool(self.encoder(tokens))          # (B, dModel)
        mlpOut = self.mlpBranch(abBlock)                   # (B, num_cells)

        out = self.fusionLayer(torch.cat([encoded, mlpOut, kFeatures], dim=-1))
        return out.reshape(*leadingShape, -1) # Omer: so leadingShape == the B,T,S .. dimensions, and basically we're returning something that is shaped back to be of size (B,T,S,action_size)
    
def loadEncoderFromSurrogate(hybridNet, checkpointPath):
    checkpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    hybridNet.encoder.load_state_dict(
        {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")},
        strict=True)
    hybridNet.pool.load_state_dict(
        {k[len("pool."):]: v for k, v in state.items() if k.startswith("pool.")},
        strict=True)


def setEncoderFrozen(hybridNet, frozen):
    for p in list(hybridNet.encoder.parameters()) + list(hybridNet.pool.parameters()):
        p.requires_grad = not frozen