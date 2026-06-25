"""
This module is a (almost) copy-past of the tutorial explained in:
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html

Some minor adjustments apply:
1. No need to use random interactions to discover the observations range if you want to normalize them, they are just 0s and 1s.
2. I needed to add a transform from int8 (multibinary) to float32.
3. Instead of Box we have a multiBinary distribution (so there is also no need or sense to probe the rnvironment for boundaries).

"""
import numpy as np
import qecc # Needed, to register bbgym with gymansium
from qecc.loggerForReinforcementLearning import logger
import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
#from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Bernoulli
from torch import nn
from torchrl.collectors import Collector as SyncDataCollector #Omer I dropped in Collector instead of SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from torchrl.envs.transforms import Transform

#myKeys = ['Observation', 'actorEntropy', 'logP',
myKeys = ['Reward', 
        'epochNumber', 
        'step_count', 
        'lr',
          "eval step count",
          "eval reward sum",
          "eval reward mean",
          "eval step count"
          ]
myLogger = logger(keys = myKeys) # The default data logging path will be grabbed in the module from a system environment variable called QECC_DATA


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
from torch.distributions import Bernoulli, Independent
class IndependentBernoulli(Independent):
    def __init__(self, logits):
        super().__init__(Bernoulli(logits=logits), 1)


is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)




num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000

total_frames = 50_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

from qecc.bb_gym import exampleDecoderFunction, exampleDecoderFunction2
print(f"Use GymEnv to wrap the environmen. Any arguments past device will be passed on to the environmet via gym.make.: ")
base_env = GymEnv("qecc/bbcode-v0", device=device, l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction2, errorRange = np.linspace(0.1,0.0001,50), minimumNumberOfLogicalQubits = 6)

print(f"Now we need to transform the observation type of multi binary which is int8, to float32 using a transformed env:")
env = TransformedEnv(
    base_env,
    Compose(
        CastToFloat(),                          # int8 → float32
        ObservationNorm(in_keys=["observation"], loc = 0.5, scale = 0.5), # loc = 0.5 and scale = 0.5 since the observation are binary. Not sure this is smart, but it would make the input to the neural network be -1 and 1 instead of 0 and 1 correspondingly
        DoubleToFloat(),
        StepCounter(),
    ),
)

#env.transform[1].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0) #Omer: Don't use this random statistical way, it's time consuming and we know what the observations look like anyway.

check_env_specs(env)

rollout = env.rollout(3)


actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(env.action_spec.shape[-1], device=device),
)

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

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
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

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for epochNumber in range(num_epochs):
        myLogger.keyValue("epochNumber", epochNumber)
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

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    myLogger.keyValue("Reward", tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    myLogger.keyValue("step_count", tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    myLogger.keyValue("lr", optim.param_groups[0]["lr"])

    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            myLogger.keyValue("eval reward mean", eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            myLogger.keyValue("eval reward sum", eval_rollout["next", "reward"].sum().item())
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            myLogger.keyValue("eval step count", eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    myLogger.dumpLogger()
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()