#!/usr/bin/env python3
"""Train a PPO agent on Balatro using Ray RLlib.

All distributed-compute knobs are exposed as CLI arguments so you can
control exactly which resources run rollout collection vs. training.

Examples
--------
# Local laptop — 4 CPU rollout workers, train on CPU
python -m agent.rllib.train --num-env-runners 4

# Single GPU training, 8 CPU rollout workers
python -m agent.rllib.train --num-env-runners 8 --num-gpus-per-learner 1

# Multi-GPU: 2 learner workers each with 1 GPU, 16 CPU rollout workers
python -m agent.rllib.train \
    --num-env-runners 16 --num-learners 2 --num-gpus-per-learner 1

# Vectorized envs on each runner (faster sampling)
python -m agent.rllib.train \
    --num-env-runners 8 --num-envs-per-env-runner 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import deque
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from agent.rllib.action_mask_model import ActionMaskingTorchRLModule
from agent.rllib.callbacks import BalatroMetricsCallback
from agent.rllib.env_wrapper import make_balatro_env
from balatro_gym.difficulty import list_difficulties


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a PPO agent on Balatro with Ray RLlib.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Environment ----
    env_group = p.add_argument_group("Environment")
    env_group.add_argument(
        "--difficulty",
        choices=list_difficulties(),
        default="easy",
        help="Game difficulty preset (any file in balatro_gym/difficulty/).",
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
    loop_group.add_argument(
        "--smoothing-window",
        type=int,
        default=20,
        help="Rolling-mean (MA) window in iterations. Used for all *_MA metrics, "
             "best-checkpoint tracking, and early stopping.",
    )
    loop_group.add_argument(
        "--early-stop-patience",
        type=int,
        default=50,
        help="Stop if reward MA hasn't improved for this many iterations. "
             "Set to a very large value (e.g. 100000) to effectively disable.",
    )
    loop_group.add_argument(
        "--early-stop-min-improvement",
        type=float,
        default=1e-8,
        help="Minimum gain in reward MA that counts as 'improvement'. "
             "Small default = lenient stopping; almost never triggers on a "
             "still-learning policy.",
    )
    loop_group.add_argument(
        "--restore-from",
        type=str,
        default=None,
        help="Path to an existing checkpoint dir to resume training from "
             "(e.g. 'checkpoints/ppo_easy_xxxx/latest'). Must match the "
             "current --difficulty and network architecture.",
    )

    # ---- Weights & Biases logging ----
    wandb_group = p.add_argument_group("Weights & Biases")
    wandb_group.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log metrics to Weights & Biases. Use --no-wandb to disable.",
    )
    wandb_group.add_argument(
        "--wandb-project",
        type=str,
        default="balatro-agent",
        help="W&B project name.",
    )
    wandb_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team or username). None = your default entity.",
    )
    wandb_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name. None = auto-generated from difficulty + timestamp.",
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


def _save_to(algo, path: Path) -> None:
    """Save the algo to an explicit path, replacing any prior contents.

    Used for ``latest/`` and ``best/`` so we never accumulate stale
    intermediate checkpoints.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    algo.save_to_path(str(path))


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
        # -- Custom metrics callback --
        .callbacks(BalatroMetricsCallback)
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

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest"
    best_ckpt = checkpoint_dir / "best"

    if args.restore_from:
        restore_path = str(Path(args.restore_from).resolve())
        print(f"Resuming from checkpoint: {restore_path}")
        algo.restore_from_path(restore_path)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            from datetime import datetime
            run_name = args.wandb_run_name or (
                f"{args.difficulty}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                dir=str(checkpoint_dir),
            )
            print(f"W&B run initialized: {wandb_run.url}")
        except ImportError:
            print("wandb not installed; skipping W&B logging. "
                  "Install with `pip install wandb` to enable.")
        except Exception as e:
            print(f"W&B initialization failed ({e}); continuing without it.")

    # Rolling-mean windows for each MA metric. All share --smoothing-window.
    MA_W = args.smoothing_window
    reward_window: deque[float]        = deque(maxlen=MA_W)
    win_rate_window: deque[float]      = deque(maxlen=MA_W)
    blinds_window: deque[float]        = deque(maxlen=MA_W)
    ante_window: deque[float]          = deque(maxlen=MA_W)
    final_score_window: deque[float]   = deque(maxlen=MA_W)
    ep_len_window: deque[float]        = deque(maxlen=MA_W)

    best_reward_MA = float("-inf")
    iters_since_improvement = 0
    metrics_history: list[dict] = []

    def _mean(window: deque) -> float:
        return sum(window) / len(window) if window else 0.0

    for i in range(1, args.num_iterations + 1):
        result = algo.train()

        # Extract standard metrics. Ray's canonical key is "episode_return_mean";
        # we just re-publish it under the friendlier name "episode_reward_mean"
        # in the wandb dict below.
        env_r = result["env_runners"]
        mean_reward = env_r.get("episode_return_mean", 0.0)
        episodes = env_r.get("num_episodes_lifetime", 0)
        # Kept for --stop-timesteps even though we no longer log it.
        timesteps = env_r.get("num_env_steps_sampled_lifetime", 0)
        ep_len_mean = env_r.get("episode_len_mean", 0.0)

        # Extract game-specific metrics from callback
        game_won = env_r.get("game_won", 0.0)
        blinds_beaten = env_r.get("blinds_beaten", 0.0)
        ante_reached = env_r.get("ante_reached", 0.0)
        final_money = env_r.get("final_money", 0.0)
        final_score = env_r.get("final_score", 0.0)

        # Update each MA window, then snapshot.
        reward_window.append(mean_reward)
        win_rate_window.append(game_won)
        blinds_window.append(blinds_beaten)
        ante_window.append(ante_reached)
        final_score_window.append(final_score)
        ep_len_window.append(ep_len_mean)

        reward_MA       = _mean(reward_window)
        win_rate_MA     = _mean(win_rate_window)
        blinds_MA       = _mean(blinds_window)
        ante_MA         = _mean(ante_window)
        final_score_MA  = _mean(final_score_window)
        ep_len_MA       = _mean(ep_len_window)

        # Metrics dict — this is exactly what gets logged to wandb.
        iter_metrics = {
            "iteration": i,
            "episodes": episodes,
            "episode_reward_mean": mean_reward,
            "episode_reward_MA":   reward_MA,
            "episode_len_mean":    ep_len_mean,
            "episode_len_MA":      ep_len_MA,
            "win_rate":            game_won,
            "win_rate_MA":         win_rate_MA,
            "blinds_beaten_mean":  blinds_beaten,
            "blinds_beaten_MA":    blinds_MA,
            "ante_reached_mean":   ante_reached,
            "ante_reached_MA":     ante_MA,
            "final_money_mean":    final_money,
            "final_score_mean":    final_score,
            "final_score_MA":      final_score_MA,
        }
        metrics_history.append(iter_metrics)

        if wandb_run is not None:
            wandb_run.log(iter_metrics, step=i)

        print(
            f"Iter {i:4d} | "
            f"reward={mean_reward:7.2f} (MA={reward_MA:7.2f}) | "
            f"win_rate={game_won:.2f} | "
            f"blinds={blinds_beaten:.1f} | "
            f"ante={ante_reached:.1f} | "
            f"ep_len={ep_len_mean:.0f}"
        )

        # Best-checkpoint tracking (uses reward MA).
        improved = reward_MA > best_reward_MA + args.early_stop_min_improvement
        if improved:
            best_reward_MA = reward_MA
            iters_since_improvement = 0
            _save_to(algo, best_ckpt)
            print(f"  -> Best checkpoint updated (reward MA {reward_MA:.3f})")
        else:
            iters_since_improvement += 1

        # Latest-checkpoint tracking.
        if i % args.checkpoint_freq == 0 or i == args.num_iterations:
            _save_to(algo, latest_ckpt)
            print(f"  -> Latest checkpoint saved")

        # Patience-based early stopping.
        if iters_since_improvement >= args.early_stop_patience:
            print(
                f"Early stop: reward MA hasn't improved by "
                f">={args.early_stop_min_improvement} for "
                f"{args.early_stop_patience} iterations."
            )
            _save_to(algo, latest_ckpt)
            break

        # Explicit user-set stop conditions.
        if args.stop_reward is not None and mean_reward >= args.stop_reward:
            print(f"Reached target reward {args.stop_reward}. Stopping.")
            _save_to(algo, latest_ckpt)
            break
        if args.stop_timesteps is not None and timesteps >= args.stop_timesteps:
            print(f"Reached {args.stop_timesteps} timesteps. Stopping.")
            _save_to(algo, latest_ckpt)
            break

    algo.stop()
    ray.shutdown()

    if wandb_run is not None:
        wandb_run.finish()

    # Save metrics history to JSON
    import json
    metrics_path = checkpoint_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    print(f"\nTraining complete. Best reward MA: {best_reward_MA:.3f}")
    if latest_ckpt.exists():
        print(f"Latest checkpoint: {latest_ckpt}")
    if best_ckpt.exists():
        print(f"Best checkpoint:   {best_ckpt}")
    return str(latest_ckpt) if latest_ckpt.exists() else None


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
