#import os
import numpy as np
import qecc # noqa: F401 — registers "qecc/bbcode-v0" with gymnasium via __init__.py # Needed, to register bbgym with gymansium
#import argparse
import warnings
#from qecc.loggerForReinforcementLearning import logger
import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Bernoulli
#from torchrl.collectors import Collector as SyncDataCollector #Omer I dropped in Collector instead of SyncDataCollector
from torchrl.collectors import MultiSyncCollector
#from torchrl.data.replay_buffers import ReplayBuffer
#from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
#from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv, ParallelEnv)
from torchrl.envs.libs.gym import GymEnv
#from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
#from torchrl.objectives import ClipPPOLoss
#from torchrl.objectives.value import GAE
#from tqdm import tqdm
from torchrl.envs.transforms import Transform
from torch.distributions import Bernoulli, Independent
from qecc.bb_gym import exampleDecoderFunction2
from qecc.modelArchitectures import create_actor_value_nets, create_value_net
warnings.filterwarnings("ignore")

env_level_paralleism = 2
minimum_number_of_qubits = 6
reward_engineering = True


class IndependentBernoulli(Independent):
    def __init__(self, logits):
        super().__init__(Bernoulli(logits=logits), 1)


class CastToFloat(Transform):
    def __init__(self):
        super().__init__(in_keys=["observation"], out_keys=["observation"])
    
    def _apply_transform(self, obs):
        return obs.to(torch.float32)
    
    def transform_observation_spec(self, observation_spec):
        observation_spec["observation"] = observation_spec["observation"].to(torch.float32)
        return observation_spec


def environmentCreatorForParallelEnv():
    
    base_env = GymEnv("qecc/bbcode-v0", l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction2, errorRange = np.linspace(0.0001,0.1,5), minimumNumberOfLogicalQubits = minimum_number_of_qubits, rewardEngineering = reward_engineering)  # removed device = device, since this will run on the CPU always

    #print(f"Now we need to transform the observation type of multi binary which is int8, to float32 using a transformed env:")
    env = TransformedEnv(
        base_env,
        Compose(
            CastToFloat(),                          # int8 → float32
            ObservationNorm(in_keys=["observation"], loc = -1.0, scale = 2.0), # loc = -1.5 and scale = 2.0 since the observation are binary. Not sure this is smart, but it would make the input to the neural network be -1 and 1 instead of 0 and 1 correspondingly
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    return env

def environmentCreatorForCollector():
    return ParallelEnv(env_level_paralleism, environmentCreatorForParallelEnv)
    


#python ~/qecc/src/qecc/reinforcementLearning.py --num-workers 4 --env-level-parallelism 2 --eval-rollout-length 1 --frames-per-batch 1 --scaling-factor 1
def test_seedingParallelEnvironment():
    env_level_paralleism = 2
    minimum_number_of_qubits = 6
    reward_engineering = True

    
    class CastToFloat(Transform):
        def __init__(self):
            super().__init__(in_keys=["observation"], out_keys=["observation"])
        
        def _apply_transform(self, obs):
            return obs.to(torch.float32)
        
        def transform_observation_spec(self, observation_spec):
            observation_spec["observation"] = observation_spec["observation"].to(torch.float32)
            return observation_spec
    


    def environmentCreatorForParallelEnv():
        
        base_env = GymEnv("qecc/bbcode-v0", l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction2, errorRange = np.linspace(0.0001,0.1,5), minimumNumberOfLogicalQubits = minimum_number_of_qubits, rewardEngineering = reward_engineering)  # removed device = device, since this will run on the CPU always

        #print(f"Now we need to transform the observation type of multi binary which is int8, to float32 using a transformed env:")
        env = TransformedEnv(
            base_env,
            Compose(
                CastToFloat(),                          # int8 → float32
                ObservationNorm(in_keys=["observation"], loc = -1.0, scale = 2.0), # loc = -1.5 and scale = 2.0 since the observation are binary. Not sure this is smart, but it would make the input to the neural network be -1 and 1 instead of 0 and 1 correspondingly
                DoubleToFloat(),
                StepCounter(),
            ),
        )
        return env

    def environmentCreatorForCollector():
        return ParallelEnv(env_level_paralleism, environmentCreatorForParallelEnv)
    
    env = ParallelEnv(env_level_paralleism, environmentCreatorForParallelEnv)

    env.set_seed(7134066)
    seeds = env.env.getSeed()
    assert seeds[0] == 7134066, f"Expected seed 7134066 but got {seeds[0]}"
    


def test_seedingMultiSyncCollector():
    
    number_of_collectors = 2
    num_cells = 256
    
    


    env = ParallelEnv(env_level_paralleism, environmentCreatorForParallelEnv)

    actor_net = create_actor_value_nets(env.action_spec, num_cells) # removed device selecting leave it to the collector
    value_net = create_value_net(num_cells)
    
    

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["logits"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["logits"],
        #distribution_class=Bernoulli,
        distribution_class=IndependentBernoulli, #Omer: note the change here. This is because we nee log_prob to be a single number, and a Bernoulli distribution returns one log_prob per coordinate.
        return_log_prob=True,
    )

    
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )


    policy_module(env.reset()) # MISLEADING ! - in the original tutorial this was done as part of a "sanity check": print("Running policy:", policy_module(env.reset())) But actually it is required to initialize the lazy linear layer.
    value_module(env.reset()) # MISLEADING ! - in the original tutorial this was done as part of a "sanity check": print("Running value:", value_module(env.reset())) But actually it is required to initialize the lazy linear layer.


    
    collector = MultiSyncCollector(
        create_env_fn= [environmentCreatorForCollector] * number_of_collectors,
        policy = policy_module,
        frames_per_batch=2,
        total_frames=2,
        split_trajs=False,
    )
    collector.set_seed(7134066) # ATTENTION ! : this is necessary to make sure we're not just running the same environment with the same seed in all parallel environments. 
    
if __name__ == "__main__":
    test_seedingMultiSyncCollector()
    #test_seedingParallelEnvironment()