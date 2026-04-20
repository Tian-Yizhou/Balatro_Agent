#!/usr/bin/env python3
"""Train a PPO agent on Balatro using Ray RLlib.

All distributed-compute knobs are exposed as CLI arguments so you can
control exactly which resources run rollout collection vs. training.

Examples
--------
# Local laptop — 4 CPU rollout workers, train on CPU
python -m balatro_gym.rllib.train --num-env-runners 4

# Single GPU training, 8 CPU rollout workers
python -m balatro_gym.rllib.train --num-env-runners 8 --num-gpus-per-learner 1

# Multi-GPU: 2 learner workers each with 1 GPU, 16 CPU rollout workers
python -m balatro_gym.rllib.train \
    --num-env-runners 16 --num-learners 2 --num-gpus-per-learner 1

# Vectorized envs on each runner (faster sampling)
python -m balatro_gym.rllib.train \
    --num-env-runners 8 --num-envs-per-env-runner 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from balatro_gym.rllib.action_mask_model import ActionMaskingTorchRLModule
from balatro_gym.rllib.env_wrapper import make_balatro_env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a PPO agent on Balatro with Ray RLlib.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Environment ----
    env_group = p.add_argument_group("Environment")
    env_group.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Game difficulty preset.",
    )
    env_group.add_argument(
        "--seed", type=int, default=None,
        help="Fixed game seed (None = random each episode).",
    )

    # ---- Distributed / resource allocation ----
    dist_group = p.add_argument_group("Distributed resources")
    dist_group.add_argument(
        "--num-env-runners",
        type=int,
        default=2,
        help="Number of parallel rollout workers (CPU). "
             "Set to 0 for single-process debugging.",
    )
    dist_group.add_argument(
        "--num-envs-per-env-runner",
        type=int,
        default=1,
        help="Vectorized envs per rollout worker.",
    )
    dist_group.add_argument(
        "--num-cpus-per-env-runner",
        type=int,
        default=1,
        help="CPUs allocated per rollout worker.",
    )
    dist_group.add_argument(
        "--num-gpus-per-env-runner",
        type=float,
        default=0,
        help="GPUs per rollout worker (usually 0).",
    )
    dist_group.add_argument(
        "--num-learners",
        type=int,
        default=0,
        help="Number of remote learner workers. "
             "0 = train on the local (driver) process.",
    )
    dist_group.add_argument(
        "--num-gpus-per-learner",
        type=float,
        default=0,
        help="GPUs per learner worker. Set to 1 for GPU training.",
    )
    dist_group.add_argument(
        "--num-cpus-per-learner",
        type=int,
        default=1,
        help="CPUs per learner worker.",
    )

    # ---- PPO hyperparameters ----
    ppo_group = p.add_argument_group("PPO hyperparameters")
    ppo_group.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    ppo_group.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    ppo_group.add_argument("--lambda-gae", type=float, default=0.95, help="GAE lambda.")
    ppo_group.add_argument("--clip-param", type=float, default=0.2, help="PPO clip parameter.")
    ppo_group.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy bonus coefficient.")
    ppo_group.add_argument("--vf-loss-coeff", type=float, default=0.5, help="Value function loss coefficient.")
    ppo_group.add_argument("--train-batch-size", type=int, default=4000, help="Total batch size per training iteration.")
    ppo_group.add_argument("--sgd-minibatch-size", type=int, default=256, help="Minibatch size for SGD updates.")
    ppo_group.add_argument("--num-epochs", type=int, default=10, help="SGD epochs per training iteration.")

    # ---- Network architecture ----
    net_group = p.add_argument_group("Network architecture")
    net_group.add_argument(
        "--fcnet-hiddens",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes for the policy/value MLP.",
    )
    net_group.add_argument(
        "--fcnet-activation",
        default="relu",
        help="Activation function (relu, tanh, etc.).",
    )

    # ---- Training loop ----
    loop_group = p.add_argument_group("Training loop")
    loop_group.add_argument(
        "--num-iterations",
        type=int,
        default=200,
        help="Number of training iterations.",
    )
    loop_group.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20,
        help="Save a checkpoint every N iterations.",
    )
    loop_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/balatro_ppo",
        help="Directory for saving checkpoints.",
    )
    loop_group.add_argument(
        "--stop-reward",
        type=float,
        default=None,
        help="Stop training when mean episode reward reaches this value.",
    )
    loop_group.add_argument(
        "--stop-timesteps",
        type=int,
        default=None,
        help="Stop training after this many environment timesteps.",
    )

    # ---- Ray ----
    ray_group = p.add_argument_group("Ray init")
    ray_group.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address. None = start local cluster.",
    )
    ray_group.add_argument(
        "--ray-num-cpus",
        type=int,
        default=None,
        help="Override total CPUs visible to Ray (local mode).",
    )
    ray_group.add_argument(
        "--ray-num-gpus",
        type=int,
        default=None,
        help="Override total GPUs visible to Ray (local mode).",
    )

    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> PPOConfig:
    """Construct a ``PPOConfig`` from parsed CLI arguments."""
    config = (
        PPOConfig()
        # -- Environment --
        .environment(
            env="Balatro",
            env_config={
                "difficulty": args.difficulty,
                "seed": args.seed,
            },
        )
        # -- Rollout workers (sampling) --
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_env_runner,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            num_gpus_per_env_runner=args.num_gpus_per_env_runner,
        )
        # -- Learner workers (training) --
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus_per_learner,
            num_cpus_per_learner=args.num_cpus_per_learner,
        )
        # -- PPO hyperparameters --
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_gae,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            vf_loss_coeff=args.vf_loss_coeff,
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.sgd_minibatch_size,
            num_epochs=args.num_epochs,
        )
        # -- Action masking RL module --
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=ActionMaskingTorchRLModule,
                model_config={
                    "head_fcnet_hiddens": args.fcnet_hiddens,
                    "head_fcnet_activation": args.fcnet_activation,
                },
            ),
        )
    )
    return config


def train(args: argparse.Namespace) -> str | None:
    """Run the training loop. Returns the path to the final checkpoint."""
    from ray.tune.registry import register_env

    ray.init(
        address=args.ray_address,
        num_cpus=args.ray_num_cpus,
        num_gpus=args.ray_num_gpus,
    )

    register_env("Balatro", make_balatro_env)

    config = build_config(args)
    algo = config.build()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_reward = float("-inf")
    final_checkpoint = None

    for i in range(1, args.num_iterations + 1):
        result = algo.train()

        mean_reward = result["env_runners"]["episode_reward_mean"]
        episodes = result["env_runners"]["num_episodes_lifetime"]
        timesteps = result["env_runners"]["num_env_steps_sampled_lifetime"]

        print(
            f"Iter {i:4d} | "
            f"reward_mean={mean_reward:8.2f} | "
            f"episodes={episodes} | "
            f"timesteps={timesteps}"
        )

        # Checkpoint
        if i % args.checkpoint_freq == 0 or i == args.num_iterations:
            save_path = algo.save(str(checkpoint_dir))
            final_checkpoint = save_path
            print(f"  -> Checkpoint saved: {save_path}")

        if mean_reward > best_reward:
            best_reward = mean_reward

        # Early stopping
        if args.stop_reward is not None and mean_reward >= args.stop_reward:
            print(f"Reached target reward {args.stop_reward}. Stopping.")
            final_checkpoint = algo.save(str(checkpoint_dir))
            break
        if args.stop_timesteps is not None and timesteps >= args.stop_timesteps:
            print(f"Reached {args.stop_timesteps} timesteps. Stopping.")
            final_checkpoint = algo.save(str(checkpoint_dir))
            break

    algo.stop()
    ray.shutdown()

    print(f"\nTraining complete. Best mean reward: {best_reward:.2f}")
    if final_checkpoint:
        print(f"Final checkpoint: {final_checkpoint}")
    return final_checkpoint


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
