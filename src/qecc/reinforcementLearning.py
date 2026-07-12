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
import qecc # noqa: F401 — registers "qecc/bbcode-ldpc-v0" with gymnasium via __init__.py # Needed, to register bbgym with gymansium
import argparse

import warnings
from datetime import datetime
from qecc.loggerForReinforcementLearning import logger
import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Bernoulli, OneHotCategorical
from torchrl.collectors import Collector as SyncDataCollector #Omer I dropped in Collector instead of SyncDataCollector

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
from qecc.modelArchitectures import create_actor_value_nets, create_value_net, hybridNet, setEncoderFrozen
warnings.filterwarnings("ignore")

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
                    "policy entropy",
                    "Encoder freeze",
                    ]




class CastToFloat(Transform):
    def __init__(self):
        super().__init__(in_keys=["observation"], out_keys=["observation"])
    
    def _apply_transform(self, obs):
        return obs.to(torch.float32)
    
    def transform_observation_spec(self, observation_spec):
        observation_spec["observation"] = observation_spec["observation"].to(torch.float32)
        return observation_spec

# The following class solves the following problem: TorchRl has a onehot distribution, and we need to stack 4 of them. 
# If they were homogenous (say each is of length K), we could use oneHot(4,K).
# However, we have 2 oneHot vectors of length l+1 and 2 of length m+1, and not necessarily l==m
# The other problem it solves is the same reason we wrapped the Bernouli distribution - we need to feed ppo with a single number for entropy

class ConcatenatedOneHotCategorical(torch.distributions.Distribution):
    """The flat logits vector is split into blocks of sizes blockSizes, e.g. (l+1, l+1, m+1, m+1).
    Each block is an independent categorical choice sampled as a one-hot vector, and the action is
    the flat concatenation of the blocks. 
    
    The step of bb_gym_v_0_1 slices the same blocks and XORs each one (minus its trailing no-op bit) into the
    polynomial representation, so each step flips at most one coefficient per polynomial.
    log_prob and entropy sum over the blocks, so log_prob is a single number, as PPO needs."""

    

    def __init__(self, logits, blockSizes):
        self.blockSizes = list(blockSizes)
        # Instantiate as many oneHot distributions as there are blocks. 
        self.blockDistributions = [OneHotCategorical(logits=blockLogits)
                                   for blockLogits in logits.split(self.blockSizes, dim=-1)]
        super().__init__(batch_shape=logits.shape[:-1], event_shape=logits.shape[-1:],
                         validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        # We sample from each oneHot distribution and return the result.
        return torch.cat([d.sample(sample_shape) for d in self.blockDistributions], dim=-1)

    def log_prob(self, value):
        blockValues = value.split(self.blockSizes, dim=-1)
        return sum(d.log_prob(v) for d, v in zip(self.blockDistributions, blockValues))

    def entropy(self):
        return sum(d.entropy() for d in self.blockDistributions)

    # We need this property to sample deterministically during evaluation which calls deterministic_sample
    @property
    def deterministic_sample(self):
        return self.mode
    
    # Deterministic sample - each oneHot distribution returns its "mode" which is the most probable coordinate 
    @property
    def mode(self):
        return torch.cat([d.mode for d in self.blockDistributions], dim=-1)





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
        "--eval-rollout-length", type=int, default=1,
        help="Length of the evaluation rollout.",
    )
    parser.add_argument(
        "--scaling-factor", type=int, default=1, #SCALING_FACTOR = 1 # Use 1 for logger testing
        help="Scaling factor for total frames, which is calculated as: total_frames = frames_per_batch * scaling_factor. This is so total_frames / frames_per_batch is an integer. Use 1 for logger testing, or checking that everything works. "
    )
    parser.add_argument(
        "--frames-per-batch", type=int, default=10, #frames_per_batch = 100 * SCALING_FACTOR
        help="Number of frames to collect per batch. Use 1000 (or bigger) for actuall runs, or 10 for testing. This also determines total_frames since total_frames = frames_per_batch * scaling_factor."
    )
    parser.add_argument(
        "--sub-batch-size", type=int, default=5, #sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
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
        "--env-minimum-number-of-qubits", type=int, default=1,
        help="Number of logical qubits from which the reward will be calculated as a code-decoder evaluation. If the code has fewer qubits, say k, the reward will exp(k-minimum_number_of_qubits). If the code has more qubits, say k, the reward will be (k-minimum_number_of_qubits)/minimum_number_of_qubits.",
    )

    parser.add_argument(
        "--log-name", type=str, default=None,
        help="Name of the log file. All logs are saved in the directory specified by the environment variable QECC_DATA. If not specified, the file name will be experiment.txt.",
    )

    parser.add_argument(
        "--env-version", type=str, default="qecc/bbcode-ldpc-v0", choices=["qecc/bbcode-v0", "qecc/bbcode-ldpc-v0"],
        help="Name of the environment to use. qecc/bbcode-bitflip-v1_0 uses a MultiDiscrete action space, while qecc/bbcode-ldpc-v0 uses MultiBinary. This will change over time as the environments evolve.",
    )

    parser.add_argument(
        "--env-bit-flipping", type=str, default="False", choices = ["True", "False", "true", "false"],
        help="Whether the agent changes (at most) one element of each polynomial at each step, or can change the entire polynommial (stateless environment).ame of the environment to use. qecc/bbcode-bitflip-v1_0 uses a MultiDiscrete action space, while qecc/bbcode-ldpc-v0 uses MultiBinary.",
    )

    parser.add_argument(
        "--env-l", type=int, default=6, choices = [6,9,15,12,30,21],
        help="parameter l for bb code construction. Can be any integer, but for now I limited the choices.",
    )

    parser.add_argument(
        "--env-m", type=int, default=6, choices = [3,6,12,18],
        help="parameter m for bb code construction. Can be any integer, but for now I limited the choices.",
    )
    
    parser.add_argument(
        "--env-use-dict-observation", default = "False", type = str, choices = ["True", "False", "true", "false"],
        help = "In any case the environment returns the observation as a flat vector containing the code. If this flag is True, then it also returns the vectors that represent the polynomial exponents so np.concat([aX,aY,bX,bY,code]). WARNING: note that this ordering is not the same as the action slices: action=  np.concat([aX,bX,aY,bY])"
    )

    parser.add_argument(
        "--model-architecture", default = "mlp", type = str, choices = ["mlp", "hybrid"],
        help = "Use mlp for the first version, where we only expose the code as a flat vector. For now, this parameter is overrdien internally to mirror --env-preamble-polynomilas-to-observation."
    )

    parser.add_argument(
        "--model-surrogate-model-path", default = None, type = str,
        help = "Path to saved surrogate model for the code-encoder to use."
    )


    parser.add_argument("--index-to-unfreeze-encoder-updates", type=int, default=10,
                        help="Number of collector batches during which the pretrained encoders stay frozen.")
    
    parser.add_argument("--encoder-lr-factor", type=float, default=0.1,
                        help="Learning-rate multiplier for the encoder+pool parameter group after unfreezing.")
    
    ## Parse the arguments, and rename some flags as local variables in a different way.
    parsedArguments = parser.parse_args()
    
    
    
    
    
    scaling_factor = parsedArguments.scaling_factor
    frames_per_batch = parsedArguments.frames_per_batch
    total_frames = frames_per_batch * scaling_factor
    max_grad_norm = parsedArguments.max_grad_norm
    sub_batch_size = parsedArguments.sub_batch_size
    num_epochs = parsedArguments.num_epochs
    eval_rollout_length = parsedArguments.eval_rollout_length
    lr = parsedArguments.lr
    num_cells = parsedArguments.num_cells
    log_name = parsedArguments.log_name
    clip_epsilon = (
        parsedArguments.clip_epsilon
        )
    gamma = parsedArguments.gamma
    lmbda = parsedArguments.lmbda
    entropy_eps = parsedArguments.entropy_eps
    num_workers = parsedArguments.num_workers
    
    seed_for_environment = parsedArguments.seed_for_environment
    
    env_reward_engineering = parsedArguments.reward_engineering.lower() == "true"
    env_bit_flipping = parsedArguments.env_bit_flipping.lower() == "true"
    env_version = parsedArguments.env_version
    #num_gpus = parsedArguments.num_gpus
    env_l = parsedArguments.env_l
    env_m = parsedArguments.env_m
    env_useDictObservation = parsedArguments.env_use_dict_observation.lower() == "true"
    env_level_paralleism = num_workers #parsedArguments.env_level_parallelism
    env_minimum_number_of_qubits = parsedArguments.env_minimum_number_of_qubits

    model_architecture = parsedArguments.model_architecture

    
    model_path_to_take_surrogate = parsedArguments.model_surrogate_model_path
    
    indexToUnfreezeEncoderUpdates = parsedArguments.index_to_unfreeze_encoder_updates
    encoder_lr_factor = parsedArguments.encoder_lr_factor
    cudaDeviceNames = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    
    device = torch.device("cpu") # Omer: right now everything is CPU bound.

    # Check conflicting definitions
    if frames_per_batch // sub_batch_size == 0:
        raise ValueError(f"frames_per_batch == {frames_per_batch} and sub_batch_size == {sub_batch_size} which means frames_per_batch // sub_batch_size == 0, not a valid configuration.")
    
    
    if env_useDictObservation:
        model_architecture = "hybrid"
    else:
        model_architecture = "mlp"
    
    
    # if num_gpus > 0:
    #     # If GPUS are provided, then the number of collectors will be equal to the number of GPUs, and each collector will be assigned to a different GPU. 
    #     num_collectors = num_gpus
    #     collectorDevices = cudaDeviceNames[:num_gpus]
    #     device = torch.device(0)
    #     #if num_workers != env_level_paralleism:
    #     #    raise ValueError("If GPUS are provided, then the number of workers should be the same as environment level parallelism")
    # else:
    #     num_collectors = 1 #max(1, num_workers // env_level_paralleism)
    #     device = torch.device("cpu")
    #     collectorDevices = [device] * num_collectors        
    
   
    if log_name is not None:
        myLogger = logger(keys = myEvaluationKeys, fileName=log_name) # The default data logging path will be grabbed in the module from a system environment variable called QECC_DATA
    else:
        myLogger = logger(keys = myEvaluationKeys) 
    
    # Dump all the flags and arguments into the log file as a comment at the top
    [myLogger.addComment(f"{key} = {value}") for key, value in vars(parsedArguments).items()]
    # Some more data about the box / machine on which this training runs:
    if os.environ.get("SLURM_CPUS_PER_TASK") is not None:
        myLogger.addComment(f"Just for information, not used in actual run: SLURM CPUS queried from os environment: {os.environ.get('SLURM_CPUS_PER_TASK')}")    
    myLogger.addComment(f"Number of workers: {num_workers}")
    myLogger.addComment(f"Does torch identify cuda: {torch.cuda.is_available()}")
    
    
    
    
    def environmentCreatorForParallelEnv():
        #print(f"Use GymEnv to wrap the environmen. Any arguments past device will be passed on to the environmet via gym.make.: ")        
        env = GymEnv("qecc/bbcode-ldpc-v0", 
                          l = env_l, 
                          m = env_m, 
                          errorRange = np.linspace(0.0001,0.1,5), 
                          minimumNumberOfLogicalQubits = env_minimum_number_of_qubits, 
                          rewardEngineering = env_reward_engineering, 
                          bitFlipping = env_bit_flipping, 
                          useDictObservation = env_useDictObservation)  # removed device = device, since this will run on the CPU always
        # If we're not taking care of it inside the model, then we need to transform the observation type of multi binary which is int8, to float32 using a transformed env:")
        # We also need to change 0,1 to -1,1 if this is not taken cared of inside the model
        if env_useDictObservation: # WARNING - if env_useDictObservation is True, that means that normalization has to happen inside the model forward method.
            env = TransformedEnv(env, Compose(StepCounter()))
                    #CastToFloat(keys=["aX", "bX", "aY", "bY", "code"]),  # int8 -> float32; "k" is already float32 from its Box space),                          # int8 → float32
                    #DoubleToFloat(),
        else: #Mode where we only spit out the flat code
            env = TransformedEnv(
                env,
                Compose(
                    CastToFloat(),                          # int8 → float32
                    ObservationNorm(in_keys=["observation"], loc = -1.0, scale = 2.0), # loc = -1.5 and scale = 2.0 since the observation are binary. Not sure this is smart, but it would make the input to the neural network be -1 and 1 instead of 0 and 1 correspondingly
                    DoubleToFloat(),
                    StepCounter(),
                ),
            )
        return env

    #def environmentCreatorForCollector():
    #    return ParallelEnv(env_level_paralleism, environmentCreatorForParallelEnv)
    
    collectorEnv = ParallelEnv(env_level_paralleism, environmentCreatorForParallelEnv, num_threads = 1)
    collectorEnv.set_seed(seed_for_environment) # ATTENTION ! : this is necessary to make sure we're not just running the same environment with the same seed in all parallel environments. 
    evaluationEnv = environmentCreatorForParallelEnv()
    evaluationEnv.set_seed(seed_for_environment + 1) # ATTENTION ! : this is necessary to make sure we're not just running the same environment with the same seed in all parallel environments. 

    if model_architecture == "mlp":
        actor_net = create_actor_value_nets(collectorEnv.action_spec, num_cells) # removed device selecting leave it to the collector
        value_net = create_value_net(num_cells)
    elif model_architecture == "hybrid":
        actor_net = hybridNet(env_l, env_m, 
                              minimumNumberOfQubits=env_minimum_number_of_qubits, # OMER: This is not a bug ! It is an argument for the env, but the actor and critic are also aware of it.
                              surrogateModelPath = model_path_to_take_surrogate,
                              outputSize = collectorEnv.action_spec.shape[-1], 
                              num_cells=num_cells, 
                              device=device)
        value_net = hybridNet(env_l, env_m, 
                              minimumNumberOfQubits = env_minimum_number_of_qubits, # OMER: This is not a bug ! It is an argument for the env, but the actor and critic are also aware of it.
                              surrogateModelPath = model_path_to_take_surrogate,
                              outputSize = 1, # The value function outputs just a scalar.
                              num_cells=num_cells, 
                              device=device)
        setEncoderFrozen(actor_net, True)
        setEncoderFrozen(value_net, True)
    else:
        raise ValueError(f"Expected the model architecture to be mlp or hybrid, instead got {model_architecture}.")
    
    

    if model_architecture == "hybrid":
        # WARNING ! There is a positional bound to
        # hybridNet.forward(self, aX, bX, aY, bY, code, numberOfLogicalQubits).
        # This list's order must equal the forward signature's order. If you reorder one, reorder the other. So the names don't make a differ(Names don't bind — position does.)
        policy_module = TensorDictModule(
            actor_net, in_keys=["aX", "bX", "aY", "bY", "code", "k"], out_keys=["logits"]
        )
        value_module = ValueOperator(
            module=value_net,
            in_keys=["aX", "bX", "aY", "bY", "code", "k"],   # same ORDER CONTRACT as policy_module
        )
    else:
        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["logits"]
        )
        value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )


    if env_bit_flipping == True:
            policy_module = ProbabilisticActor(
            module=policy_module,
            spec=collectorEnv.action_spec,
            in_keys=["logits"],
            #distribution_class=Bernoulli,
            distribution_class= ConcatenatedOneHotCategorical, #Omer: note that we're using a class at the top of this module which extends the onehot distribution.
            distribution_kwargs={"blockSizes": [env_l + 1, env_l + 1, env_m + 1, env_m + 1]},
            return_log_prob=True,
        )
    elif env_bit_flipping == False:
            policy_module = ProbabilisticActor(
            module=policy_module,
            spec=collectorEnv.action_spec,
            in_keys=["logits"],
            #distribution_class=Bernoulli,
            distribution_class=IndependentBernoulli, #Omer: note the change here. This is because we need log_prob to be a single number, and a Bernoulli distribution returns one log_prob per coordinate.
            return_log_prob=True,
        )
    else:
        raise ValueError("The value of env_bit_flipping can be True or False but is {env_bit_flipping}.")

    
    # Omer: Potentially we don't need this anymore once we switch to non-lazy linear.
    policy_module(collectorEnv.reset()) # MISLEADING ! - in the original tutorial this was done as part of a "sanity check": print("Running policy:", policy_module(env.reset())) But actually it is required to initialize the lazy linear layer.
    value_module(collectorEnv.reset()) # MISLEADING ! - in the original tutorial this was done as part of a "sanity check": print("Running value:", value_module(env.reset())) But actually it is required to initialize the lazy linear layer.

    
   
   
    collector = SyncDataCollector(
        collectorEnv,
        policy = policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
        storing_device = device
    )
    

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

    if model_architecture == "hybrid":
        # We separate the optimization to encoder parameters and everything else:
        encoderParameters = (list(actor_net.encoder.parameters()) + list(actor_net.pool.parameters())
                             + list(value_net.encoder.parameters()) + list(value_net.pool.parameters()))
        encoderParameterIds = {id(parameter) for parameter in encoderParameters}
        otherParameters = [parameter for parameter in loss_module.parameters()
                          if id(parameter) not in encoderParameterIds]
        optim = torch.optim.Adam([
            {"params": otherParameters,    "lr": lr},
            {"params": encoderParameters, "lr": lr * encoder_lr_factor},
        ])
    else:
        optim = torch.optim.Adam(loss_module.parameters(), lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    #logs = defaultdict(list)
    print("Training begins on:")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S %A"))
    pbar = tqdm(total=total_frames)
    eval_str = ""
    # Flag to report whether encoder weights are frozen or not 
    isPretrainedEncoderFrozen = True
    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:

    for i, tensordict_data in enumerate(collector):
        
        if model_architecture == "hybrid" and i == indexToUnfreezeEncoderUpdates:
            setEncoderFrozen(actor_net, False)
            setEncoderFrozen(value_net, False)
            isPretrainedEncoderFrozen = False
            #myLogger.addComment(f"Unfroze pretrained encoders at collector batch {i}; encoder lr = {lr * encoder_lr_factor}.")
        
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
                eval_policy = policy_module
                # if num_gpus > 0:
                #     eval_policy = copy.deepcopy(policy_module).to("cpu")
                # execute a rollout with the trained policy        
                eval_rollout = evaluationEnv.rollout(eval_rollout_length, eval_policy) 
                #logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                dist = eval_policy.get_dist(eval_rollout)
                entropiesDuringEvaluation = dist.entropy().cpu().numpy()
                rewards = eval_rollout["next", "reward"].cpu().numpy() 
            
                for timeIndex in range(eval_rollout_length):
                    #myLogger.keyValue(f"environment index", envIndex)
                    myLogger.keyValue("evaluation number", i // 10)  # i is not epochNumber, but this is purely for debug puposes.
                    #myLogger.keyValue("observation", eval_rollout["observation"].cpu().numpy()[k])
                    #myLogger.keyValue("action", eval_rollout["action"].cpu().numpy())
                    myLogger.keyValue("reward", rewards[timeIndex].item())
                    myLogger.keyValue("policy entropy", entropiesDuringEvaluation[timeIndex].item())
                    myLogger.keyValue("Encoder freeze", isPretrainedEncoderFrozen)
                    myLogger.dumpLogger(printOut = False)
                
                del eval_rollout
        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    torch.save(policy_module.state_dict(), f"{myLogger.logPath}/policy_weights.pth")
    torch.save(value_module.state_dict(), f"{myLogger.logPath}/value_weights.pth")
    print(f"Finished.") 
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S %A"))
    print(f"Experiment logs and policy weights are located in:\n{myLogger.logPath}")


