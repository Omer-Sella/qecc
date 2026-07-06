"""
This module is a (almost) copy-past of the tutorial explained in:
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html

Some minor adjustments apply:
1. No need to use random interactions to discover the observations range if you want to normalize them, they are just 0s and 1s.
2. I needed to add a transform from int8 (multibinary) to float32.
3. Instead of Box we have a multiBinary distribution (so there is also no need or sense to probe the rnvironment for boundaries).


Multiprocessing notes:

There are A LOT of fine issues here.
TorchRL provides a parallel env class, and also a distributed data collector.
To make the most out of multiprocessing, you actually need to fine tune between them.


For seeding multiple environments, see the explanation in the following link:
https://github.com/pytorch/rl/blob/main/tutorials/sphinx-tutorials/torchrl_envs.py


For documentation on using ParallelEnv see the following link:
https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.ParallelEnv.html).
https://deepwiki.com/pytorch/rl/3.4-batched-and-parallel-environments

For a clairifcation on using ParallelEnv and multiSyncDataCollector see the following link:
https://github.com/pytorch/rl/issues/809

On data collectors:
https://deepwiki.com/pytorch/rl/4.2-distributed-collection-strategies


"""
import os

import numpy as np
import qecc # noqa: F401 — registers "qecc/bbcode-v0" with gymnasium via __init__.py # Needed, to register bbgym with gymansium
from qecc.loggerForReinforcementLearning import logger
import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing
import torch
from tensordict.nn import TensorDictModule
#from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Bernoulli
from torchrl.collectors import Collector as SyncDataCollector #Omer I dropped in Collector instead of SyncDataCollector
from torchrl.collectors import MultiSyncCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv, ParallelEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from torchrl.envs.transforms import Transform
from torch.distributions import Bernoulli, Independent
from qecc.bb_gym import exampleDecoderFunction2
import argparse
from qecc.modelArchitectures import create_actor_value_nets, create_value_net


#myKeys = ['Observation', 'actorEntropy', 'logP',
myKeys = ['Reward', 
        'epochNumber', 
        'step_count', 
        'lr',
          "eval step count",
          "eval reward sum",
          "eval reward mean",
          "eval step count",
          "eval observation",
          "eval action",
          "eval next observation",
          "eval next reward"
          ]

myEvaluationKeys = ["evaluation number",
                    "reward",
                    "policy entropy"]




class CastToFloat(Transform):
    def __init__(self):
        super().__init__(in_keys=["observation"], out_keys=["observation"])
    
    def _apply_transform(self, obs):
        return obs.to(torch.float32)
    
    def transform_observation_spec(self, observation_spec):
        observation_spec["observation"] = observation_spec["observation"].to(torch.float32)
        return observation_spec
    
# log_prob fix: When we ask a Bernoulli distribution to return log_prob, it returns an array, each element of which is a log_prob for that specific coordinate.
# However, for computing the loss we need to sum them.

class IndependentBernoulli(Independent):
    def __init__(self, logits):
        super().__init__(Bernoulli(logits=logits), 1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of CPUs for the whole run. ",
    )

    parser.add_argument(
        "--num-gpus", type=int, default=1, choices=[0, 1, 2, 4],
        help="Number of data collectors to use. Each collector will have its own environment and policy. The number of workers will be divided among the collectors.",
    )
    parser.add_argument(
        "--env-level-parallelism", type=int, default = 2,
        help="Number of parallel environments for each data collector to work on.",
    )

    parser.add_argument(
        "--num-cells", type=int, default=256,
        help="Number of cells in each layer of the actor and value networks.", #  num_cells = 256  # number of cells in each layer i.e. output dim.
    )
    
    parser.add_argument(
        "--seed-for-environment", type=int, default=7134066,
        help="Random seed for reproducibility. NOT SUPPORTED YET AS THIS IS DONE INTERNALLY", #  num_cells = 256  # number of cells in each layer i.e. output dim.
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate for the Adam optimizer.", #lr = 3e-4 # Learning rate
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, #max_grad_norm = 1.0
    )

    parser.add_argument(
        "--eval-rollout-length", type=int, default=50,
        help="Length of the evaluation rollout.",
    )
    parser.add_argument(
        "--scaling-factor", type=int, default=1, #SCALING_FACTOR = 1 # Use 1 for logger testing
        help="Scaling factor for total frames, which is calculated as: total_frames = frames_per_batch * scaling_factor. This is so total_frames / frames_per_batch is an integer. Use 1 for logger testing, or checking that everything works. "
    )
    parser.add_argument(
        "--frames-per-batch", type=int, default=100, #frames_per_batch = 100 * SCALING_FACTOR
        help="Number of frames to collect per batch. Use 1000 (or bigger) for actuall runs, or 10 for testing. This also determines total_frames since total_frames = frames_per_batch * scaling_factor."
    )
    parser.add_argument(
        "--sub-batch-size", type=int, default=64, #sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
        help="Size of the sub-batches for optimization. Will be multiplied by the scaling factor."
    )
    parser.add_argument(
        "--num-epochs", type=int, default=10, #num_epochs = 10
    )
    

    parser.add_argument(
        "--clip-epsilon", type=float, default=0.2, #clip_epsilon = 0.2  # clip value for PPO loss: see the equation in the intro for more context.
        help = "Clip value for PPO loss."
    )

    parser.add_argument(
        "--gamma", type=float, default=0.99, #gamma = 0.99
        help="Discount factor for the RL algorithm."
    )
    parser.add_argument(
        "--lmbda", type=float, default=0.95, #lmbda = 0.95
        help="Lambda parameter for GAE."
    )
    parser.add_argument(
        "--entropy-eps", type=float, default=1e-4, #entropy_eps = 1e-4
        help="Epsilon for the entropy bonus."
    )

    parser.add_argument(
        "--reward-engineering", type=str, default = "False", choices = ["True", "False", "true", "false"],
        help="Whether to exponentiate the already positive reward.",
    )

    

    parser.add_argument(
        "--minimum-number-of-qubits", type=int, default=1,
        help="Number of logical qubits from which the reward will be calculated as a code-decoder evaluation. If the code has fewer qubits, say k, the reward will exp(k-minimum_number_of_qubits). If the code has more qubits, say k, the reward will be exp(minimum_number_of_qubits-k).",
    )

    parser.add_argument(
        "--log-name", type=str, default=None,
        help="Name of the log file. All logs are saved in the directory specified by the environment variable QECC_DATA. If not specified, the file name will be experiment.txt.",
    )
    
    minimum_number_of_qubits = parser.parse_args().minimum_number_of_qubits
    seed_for_environment = parser.parse_args().seed_for_environment
    reward_engineering = parser.parse_args().reward_engineering.lower() == "true"
    scaling_factor = parser.parse_args().scaling_factor
    frames_per_batch = parser.parse_args().frames_per_batch
    total_frames = frames_per_batch * scaling_factor
    max_grad_norm = parser.parse_args().max_grad_norm
    sub_batch_size = parser.parse_args().sub_batch_size
    num_epochs = parser.parse_args().num_epochs
    eval_rollout_length = parser.parse_args().eval_rollout_length
    lr = parser.parse_args().lr
    num_cells = parser.parse_args().num_cells
    log_name = parser.parse_args().log_name
    num_workers = parser.parse_args().num_workers
    clip_epsilon = (
        parser.parse_args().clip_epsilon
        )
    gamma = parser.parse_args().gamma
    lmbda = parser.parse_args().lmbda
    entropy_eps = parser.parse_args().entropy_eps
    num_workers = parser.parse_args().num_workers
    num_gpus = parser.parse_args().num_gpus
    
    cudaDeviceNames = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    env_level_paralleism = parser.parse_args().env_level_parallelism
    if num_gpus > 0:
        num_collectors = num_gpus
        collectorDevices = cudaDeviceNames[:num_gpus]
        device = torch.device(0)
    else:
        num_collectors = max(1, num_workers // env_level_paralleism)
        device = torch.device("cpu")
        collectorDevices = [device] * num_collectors        
    
    

    #is_fork = multiprocessing.get_start_method() == "fork"
    #device = ( 
    #    torch.device(0)
    #    if torch.cuda.is_available() and not is_fork
    #    else torch.device("cpu")
    #)

    log_name = parser.parse_args().log_name
    if log_name is not None:
        myLogger = logger(keys = myEvaluationKeys, fileName=log_name) # The default data logging path will be grabbed in the module from a system environment variable called QECC_DATA
    else:
        myLogger = logger(keys = myEvaluationKeys) 
        
    [myLogger.addComment(f"{key} = {value}") for key, value in vars(parser.parse_args()).items()]
    
    if os.environ.get("SLURM_CPUS_PER_TASK") is not None:
        myLogger.addComment(f"Just for information, not used in actual run: SLURM CPUS queried from os environment: {os.environ.get('SLURM_CPUS_PER_TASK')}")
    
    myLogger.addComment(f"Does torch identify cuda: {torch.cuda.is_available()}")

    #print(f"Use GymEnv to wrap the environmen. Any arguments past device will be passed on to the environmet via gym.make.: ")
    
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
        create_env_fn= [environmentCreatorForCollector] * num_collectors,
        policy = policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=collectorDevices,
    )
    collector.set_seed(seed_for_environment) # ATTENTION ! : this is necessary to make sure we're not just running the same environment with the same seed in all parallel environments. 

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coeff=entropy_eps, #Note the move from coef to coeff
        # these keys match by default but we set this for completeness
        critic_coeff=1.0, # Note the move from coef to coeff
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    #logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for epochNumber in range(num_epochs):
            # myLogger.keyValue("epochNumber", epochNumber)
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        pbar.update(tensordict_data.numel())
        
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy        
                eval_rollout = env.rollout(eval_rollout_length, policy_module) 
                #logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                dist = policy_module.get_dist(eval_rollout)
                entropyDuringEvaluation = dist.entropy().cpu().numpy()
                for k in range(eval_rollout_length):
                    myLogger.keyValue("evaluation number", i // 10)  # i is not epochNumber, but this is purely for debug puposes.
                    #myLogger.keyValue("observation", eval_rollout["observation"].cpu().numpy()[k])
                    #myLogger.keyValue("action", eval_rollout["action"].cpu().numpy())
                    myLogger.keyValue("reward", eval_rollout["next", "reward"].cpu().numpy()[k].item())
                    myLogger.keyValue("policy entropy", entropyDuringEvaluation[k].item())
                    myLogger.dumpLogger(printOut = False)
                torch.save(policy_module.state_dict(), f"{myLogger.logPath}/evaluation_number_{i // 10}_policy_weights.pth")
                torch.save(value_module.state_dict(), f"{myLogger.logPath}/evaluation_number_{i // 10}_value_weights.pth")
                del eval_rollout
        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    torch.save(policy_module.state_dict(), f"{myLogger.logPath}/policy_weights.pth")
    torch.save(value_module.state_dict(), f"{myLogger.logPath}/value_weights.pth")


