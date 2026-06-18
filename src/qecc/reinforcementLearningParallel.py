"""
Parallel-environment version of reinforcementLearning.py, for running PPO on an
HPC node with multiple CPUs (single node, multiple cores).

This module is an augmentation of the tutorial explained in:
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html

With several adjustments:
1. No need to use random interactions to discover the observations range if you want to normalize them, they are just 0s and 1s.
2. I needed to add a transform from int8 (multibinary) to float32.
3. Instead of Box we have a multiBinary distribution (so there is also no need or sense to probe the rnvironment for boundaries).

This file does not work for Windows environments, 
because the ParallelEnv class uses multiprocessing with the "spawn" start method on Windows. 
On Linux, "fork" start method supports shared memory, 
which is required for the ParallelEnv to work correctly.


This file does NOT import from reinforcementLearning.py: that module runs its
whole training loop at import time (no `if __name__ == "__main__":` guard), so
importing it here would re-run the PoC as a side effect, and would do so again
inside every worker process under multiprocessing's "spawn" start method. The
two small helper classes it defines (CastToFloat, IndependentBernoulli) are
therefore duplicated below instead.

Parallelism is done via torchrl.envs.ParallelEnv, which runs num_workers copies
of the environment in worker subprocesses and exposes them as a single batched
env. 
A single SyncDataCollector on top of that collects frames from all workers
each iteration. This is the standard torchrl pattern for CPU-bound, single-node,
multi-core vectorized environments (see
https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.ParallelEnv.html).


"""
import argparse
import os

import qecc  # noqa: F401 — registers "qecc/bbcode-v0" with gymnasium via __init__.py
from qecc.loggerForReinforcementLearning import logger
import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing
from collections import defaultdict
import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Bernoulli, Independent
from torch import nn
from torchrl.collectors import Collector as SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, EnvCreator, ObservationNorm,
                          ParallelEnv, StepCounter, TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import Transform
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from qecc.bb_gym import exampleDecoderFunction

myKeys = ['Reward',
        'epochNumber',
        'step_count',
        'lr',
          "eval step count",
          "eval reward sum",
          "eval reward mean",
          "eval step count"
          ]


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


num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_worker = 128  

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4


def make_env():
    """Top-level (picklable) env factory, used both for the spec-check env and as the
    per-worker constructor passed to ParallelEnv. Always runs on CPU: the underlying
    gym env is numpy-backed and has no GPU path."""
    base_env = GymEnv(
        "qecc/bbcode-v0",
        device="cpu",
        l=6,
        m=6,
        evaluationDecoderFunction=exampleDecoderFunction,
        errorRange=[0.01, 0.001],
        minimumNumberOfLogicalQubits=6,
    )
    return TransformedEnv(
        base_env,
        Compose(
            CastToFloat(),                          # int8 -> float32
            ObservationNorm(in_keys=["observation"], loc=0.5, scale=0.5),
            DoubleToFloat(),
            StepCounter(),
        ),
    )


def resolve_num_workers(cliValue):
    if cliValue is not None:
        return cliValue
    for environmentVariable in ("SLURM_CPUS_PER_TASK", "QECC_NUM_WORKERS"):
        value = os.environ.get(environmentVariable)
        if value is not None:
            return int(value)
    return os.cpu_count()


def parseArguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel environment worker processes. Defaults to "
             "$SLURM_CPUS_PER_TASK, then $QECC_NUM_WORKERS, then os.cpu_count().",
    )
    parser.add_argument(
        "--total-frames", type=int, default=1_000_000,
        help="Total number of environment frames to collect across all workers.",
    )
    parser.add_argument(
        "--eval-horizon", type=int, default=1000,
        help="Number of steps in the periodic evaluation rollout (every 10 training iterations). "
             "Reduce to a small value (e.g. 3) for smoke-testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    args = parseArguments()
    num_workers = resolve_num_workers(args.num_workers)
    total_frames = args.total_frames
    eval_horizon = args.eval_horizon
    frames_per_batch = num_workers * frames_per_worker
    if total_frames < frames_per_batch:
        raise ValueError(
            f"--total-frames ({total_frames}) must be >= frames_per_batch "
            f"({num_workers} workers × {frames_per_worker} frames/worker = {frames_per_batch})."
        )

    is_fork = multiprocessing.get_start_method() == "fork"
    policy_device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    print(f"Using {num_workers} parallel environment workers "
          f"(frames_per_batch={frames_per_batch}, total_frames={total_frames}, "
          f"policy_device={policy_device}).")

    myLogger = logger(keys=myKeys)  # Default data logging path is grabbed from the QECC_DATA system environment variable.

    print("Building a single env instance to validate specs before spawning workers:")
    check_env = make_env()
    check_env_specs(check_env)
    check_env.rollout(3)
    action_size = check_env.action_spec.shape[-1]
    check_env.close()

    print(f"Spawning {num_workers} parallel environment workers via ParallelEnv:")
    env = ParallelEnv(num_workers, EnvCreator(make_env))

    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=policy_device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=policy_device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=policy_device),
        nn.Tanh(),
        nn.LazyLinear(action_size, device=policy_device),
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["logits"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["logits"],
        distribution_class=IndependentBernoulli,  # log_prob must be a single number per step, not one per coordinate.
        return_log_prob=True,
    )

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=policy_device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=policy_device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=policy_device),
        nn.Tanh(),
        nn.LazyLinear(1, device=policy_device),
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
        device=policy_device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=policy_device,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coeff=entropy_eps,
        critic_coeff=1.0,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    try:
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
                    loss_vals = loss_module(subdata.to(policy_device))
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
                # number of steps. The ``rollout`` method of the ``env`` can take a
                # policy as argument: it will then execute this policy at each step,
                # across all parallel workers at once.
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    eval_rollout = env.rollout(eval_horizon, policy_module)
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
    finally:
        collector.shutdown()
        if not env.is_closed:
            env.close()
