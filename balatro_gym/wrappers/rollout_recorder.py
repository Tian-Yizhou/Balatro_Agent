"""Gymnasium wrapper that records per-step rollout data for offline/online RL.

Each episode is saved as a compressed ``.npz`` file containing aligned arrays
of observations, actions, rewards, and episode metadata.  Files are small
(gzip-compressed numpy arrays), uniquely named, and loadable with a single
``np.load()`` call.

Typical usage::

    from balatro_gym.wrappers import RolloutRecorder

    env = gym.make("Balatro-Easy-v0")
    env = RolloutRecorder(env, save_dir="data/rollouts")

    obs, info = env.reset(seed=42)
    # ... play episode ...
    # On termination/truncation the .npz is flushed automatically.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


class RolloutRecorder(gym.Wrapper):
    """Records full trajectories (obs, action, reward, done) per episode.

    Each episode produces one ``.npz`` file in *save_dir* with the arrays:

    ============== ======== ===============================================
    Key            dtype    Shape / description
    ============== ======== ===============================================
    ``obs``        float32  (T+1, obs_dim) — observations including final
    ``actions``    int32    (T,) — actions taken
    ``rewards``    float32  (T,) — rewards received
    ``terminated`` bool     (T,) — per-step terminated flags
    ``truncated``  bool     (T,) — per-step truncated flags
    ``phases``     uint8    (T,) — encoded game phase at each step
    ``antes``      uint8    (T,) — ante number at each step
    ``scores``     int32    (T,) — cumulative score at each step
    ``money``      int16    (T,) — money at each step
    ============== ======== ===============================================

    Files are named ``rollout_{timestamp}_{episode_id}.npz`` for uniqueness.

    Args:
        env: The Gymnasium environment to wrap.
        save_dir: Directory to write ``.npz`` files into (created if needed).
        save_action_mask: If True, also store the boolean action mask per step.
            Doubles file size but useful for offline masked-action training.
    """

    # Phase string -> uint8 for compact storage
    _PHASE_MAP = {"play": 0, "shop": 1, "game_over": 2, "game_won": 3}

    def __init__(
        self,
        env: gym.Env,
        save_dir: str | Path = "data/rollouts",
        save_action_mask: bool = False,
    ):
        super().__init__(env)
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._save_action_mask = save_action_mask

        self._episode_id: int = 0
        self._session_ts: str = time.strftime("%Y%m%d_%H%M%S")

        # Buffers (reset each episode)
        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[int] = []
        self._rew_buf: list[float] = []
        self._term_buf: list[bool] = []
        self._trunc_buf: list[bool] = []
        self._mask_buf: list[np.ndarray] = []
        self._phase_buf: list[int] = []
        self._ante_buf: list[int] = []
        self._score_buf: list[int] = []
        self._money_buf: list[int] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Flush previous episode if there is buffered data
        if self._act_buf:
            self._flush()

        obs, info = self.env.reset(seed=seed, options=options)

        # Capture episode seed ID from the info dict (set by BalatroEnv)
        self._episode_seed_id = info.get("episode_seed_id", "")

        # Start new episode buffers
        self._obs_buf = [obs.copy()]
        self._act_buf = []
        self._rew_buf = []
        self._term_buf = []
        self._trunc_buf = []
        self._mask_buf = []
        self._phase_buf = []
        self._ante_buf = []
        self._score_buf = []
        self._money_buf = []

        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Record transition
        self._act_buf.append(action)
        self._rew_buf.append(reward)
        self._term_buf.append(terminated)
        self._trunc_buf.append(truncated)
        self._obs_buf.append(obs.copy())

        if self._save_action_mask and "action_mask" in info:
            self._mask_buf.append(info["action_mask"].copy())

        # Compact per-step metadata
        self._phase_buf.append(self._PHASE_MAP.get(info.get("phase", ""), 255))
        self._ante_buf.append(info.get("ante", 0))
        self._score_buf.append(info.get("score", 0))
        self._money_buf.append(info.get("money", 0))

        # Auto-flush on episode end
        if terminated or truncated:
            self._flush()

        return obs, reward, terminated, truncated, info

    def _flush(self) -> None:
        """Write current episode buffer to a compressed .npz file."""
        if not self._act_buf:
            return

        data: dict[str, np.ndarray] = {
            "obs": np.array(self._obs_buf, dtype=np.float32),
            "actions": np.array(self._act_buf, dtype=np.int32),
            "rewards": np.array(self._rew_buf, dtype=np.float32),
            "terminated": np.array(self._term_buf, dtype=bool),
            "truncated": np.array(self._trunc_buf, dtype=bool),
            "phases": np.array(self._phase_buf, dtype=np.uint8),
            "antes": np.array(self._ante_buf, dtype=np.uint8),
            "scores": np.array(self._score_buf, dtype=np.int32),
            "money": np.array(self._money_buf, dtype=np.int16),
        }

        if self._save_action_mask and self._mask_buf:
            data["action_masks"] = np.packbits(
                np.array(self._mask_buf, dtype=np.uint8), axis=1
            )
            data["action_mask_n"] = np.array(
                [self._mask_buf[0].shape[0]], dtype=np.int32
            )

        # Store episode seed ID as a numpy string array
        if self._episode_seed_id:
            data["episode_seed_id"] = np.array([self._episode_seed_id])

        fname = f"rollout_{self._session_ts}_{self._episode_id:06d}.npz"
        path = self._save_dir / fname
        np.savez_compressed(path, **data)

        self._episode_id += 1
        self._act_buf = []

    @staticmethod
    def load(path: str | Path) -> dict[str, np.ndarray]:
        """Load a rollout file and return its arrays as a dict.

        If action masks were stored with ``packbits``, they are automatically
        unpacked back to a boolean array of the original width.

        Returns:
            Dict with keys ``obs``, ``actions``, ``rewards``, ``terminated``,
            ``truncated``, ``phases``, ``antes``, ``scores``, ``money``, and
            optionally ``action_masks``.
        """
        raw = dict(np.load(path, allow_pickle=False))
        if "action_masks" in raw and "action_mask_n" in raw:
            n = int(raw.pop("action_mask_n")[0])
            packed = raw["action_masks"]
            raw["action_masks"] = np.unpackbits(packed, axis=1)[:, :n].astype(bool)
        if "episode_seed_id" in raw:
            raw["episode_seed_id"] = str(raw["episode_seed_id"][0])
        return raw
