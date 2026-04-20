#!/usr/bin/env python3
"""Evaluate a trained PPO checkpoint on Balatro.

Loads a saved RLlib checkpoint, runs episodes, and prints aggregate stats.
Can also be used for single-episode replay / debugging.

Examples
--------
# Evaluate a checkpoint over 100 episodes
python -m balatro_gym.rllib.evaluate \
    --checkpoint checkpoints/balatro_ppo/checkpoint_000200 \
    --num-episodes 100 --difficulty easy

# GPU inference
python -m balatro_gym.rllib.evaluate \
    --checkpoint checkpoints/balatro_ppo/checkpoint_000200 \
    --num-env-runners 4 --num-gpus-per-env-runner 0
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from balatro_gym.envs.balatro_env import TOTAL_ACTIONS
from balatro_gym.envs.configs import GameConfig
from balatro_gym.rllib.action_mask_model import ActionMaskingTorchRLModule
from balatro_gym.rllib.env_wrapper import BalatroRLlibEnv, make_balatro_env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent on Balatro.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to an RLlib checkpoint directory.",
    )
    p.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes to run.",
    )
    p.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Game difficulty preset.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed game seed (None = random each episode).",
    )

    # Resource control
    dist = p.add_argument_group("Distributed resources")
    dist.add_argument(
        "--num-env-runners",
        type=int,
        default=0,
        help="Number of remote rollout workers for parallel evaluation. "
             "0 = evaluate on the driver process.",
    )
    dist.add_argument(
        "--num-gpus-per-env-runner",
        type=float,
        default=0,
        help="GPUs per evaluation rollout worker.",
    )

    # Ray init
    ray_grp = p.add_argument_group("Ray init")
    ray_grp.add_argument("--ray-address", type=str, default=None)
    ray_grp.add_argument("--ray-num-cpus", type=int, default=None)
    ray_grp.add_argument("--ray-num-gpus", type=int, default=None)

    # Output
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode results.",
    )

    return p.parse_args(argv)


def evaluate(args: argparse.Namespace) -> dict:
    """Run evaluation episodes and return aggregate stats."""
    from ray.tune.registry import register_env

    ray.init(
        address=args.ray_address,
        num_cpus=args.ray_num_cpus,
        num_gpus=args.ray_num_gpus,
    )

    register_env("Balatro", make_balatro_env)

    # Build config matching the checkpoint's settings
    config = (
        PPOConfig()
        .environment(
            env="Balatro",
            env_config={
                "difficulty": args.difficulty,
                "seed": args.seed,
            },
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_gpus_per_env_runner=args.num_gpus_per_env_runner,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=ActionMaskingTorchRLModule,
            ),
        )
    )

    algo = config.build()
    algo.restore(args.checkpoint)

    # Run episodes locally for detailed per-episode stats
    factory = {"easy": GameConfig.easy, "medium": GameConfig.medium, "hard": GameConfig.hard}
    game_config = factory[args.difficulty](seed=args.seed)

    from balatro_gym.envs.balatro_env import BalatroEnv
    inner_env = BalatroEnv(config=game_config)
    env = BalatroRLlibEnv(inner_env)

    module = algo.get_module()
    module.eval()

    import torch

    stats: dict[str, list] = defaultdict(list)

    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=(args.seed + ep) if args.seed is not None else None)
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            obs_tensor = {
                k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                for k, v in obs.items()
            }
            with torch.no_grad():
                fwd_out = module.forward_inference({"obs": obs_tensor})

            # Sample action from the masked logits
            from ray.rllib.core.columns import Columns
            logits = fwd_out[Columns.ACTION_DIST_INPUTS].squeeze(0)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

        # Collect stats
        won = info.get("won", False)
        stats["reward"].append(episode_reward)
        stats["steps"].append(episode_steps)
        stats["won"].append(won)
        stats["ante"].append(info.get("ante", 0))
        stats["blinds_beaten"].append(info.get("blinds_beaten", 0))
        stats["money"].append(info.get("money", 0))
        stats["score"].append(info.get("score", 0))

        if args.verbose:
            status = "WON" if won else "LOST"
            print(
                f"Episode {ep + 1:4d}: {status} | "
                f"reward={episode_reward:7.2f} | "
                f"steps={episode_steps:4d} | "
                f"ante={info.get('ante', '?')} | "
                f"blinds={info.get('blinds_beaten', '?')}"
            )

    algo.stop()
    ray.shutdown()

    # Aggregate
    results = {
        "num_episodes": args.num_episodes,
        "difficulty": args.difficulty,
        "win_rate": np.mean(stats["won"]),
        "mean_reward": np.mean(stats["reward"]),
        "std_reward": np.std(stats["reward"]),
        "mean_steps": np.mean(stats["steps"]),
        "mean_ante": np.mean(stats["ante"]),
        "mean_blinds_beaten": np.mean(stats["blinds_beaten"]),
        "mean_money": np.mean(stats["money"]),
        "max_score": np.max(stats["score"]) if stats["score"] else 0,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Difficulty:         {results['difficulty']}")
    print(f"  Episodes:           {results['num_episodes']}")
    print(f"  Win rate:           {results['win_rate']:.1%}")
    print(f"  Mean reward:        {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean steps:         {results['mean_steps']:.1f}")
    print(f"  Mean ante reached:  {results['mean_ante']:.2f}")
    print(f"  Mean blinds beaten: {results['mean_blinds_beaten']:.2f}")
    print(f"  Mean final money:   {results['mean_money']:.1f}")
    print(f"  Max score:          {results['max_score']}")
    print("=" * 60)

    return results


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
