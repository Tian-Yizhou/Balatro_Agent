"""Pluggable reward functions for the Balatro environment.

The environment calls ``reward_fn(ctx)`` each step, where *ctx* is a
:class:`RewardContext` containing everything needed to compute a reward.
Users can supply any callable with signature ``(RewardContext) -> float``,
or subclass :class:`RewardFunction` for a structured approach.

Built-in reward functions:

* :class:`DefaultReward` — shaped rewards with score progress, blind-beaten
  bonuses, and terminal win/lose signals (the original hardcoded design).
* :class:`SparseReward` — only +1 for winning, -1 for losing.

Examples::

    # Use the default (shaped) reward
    env = balatro_gym.make("easy")

    # Sparse reward only
    from balatro_gym.envs.rewards import SparseReward
    env = balatro_gym.make("easy", reward_fn=SparseReward())

    # Custom reward via lambda
    env = balatro_gym.make("easy", reward_fn=lambda ctx: ctx.win * 100)

    # Custom reward class
    from balatro_gym.envs.rewards import RewardFunction, RewardContext

    class MyReward(RewardFunction):
        def __call__(self, ctx: RewardContext) -> float:
            r = 0.0
            if ctx.blind_just_beaten:
                r += 5.0
            if ctx.won:
                r += 50.0
            if ctx.lost:
                r -= 10.0
            return r

    env = balatro_gym.make("easy", reward_fn=MyReward())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RewardContext:
    """Read-only snapshot of game state passed to reward functions.

    All fields needed for reward computation are provided here.
    The environment populates this each step before calling the reward fn.

    Attributes:
        phase: Current game phase string ("play", "shop", "game_over", "game_won").
        action_valid: Whether the action executed successfully (False = invalid action).
        prev_score: Score before this step.
        new_score: Score after this step.
        score_target: Current blind's target score.
        prev_blinds_beaten: Blinds beaten before this step.
        new_blinds_beaten: Blinds beaten after this step.
        total_blinds: Total number of blinds in the game.
        blind_just_beaten: Whether a blind was beaten this step.
        won: Whether the game was just won this step.
        lost: Whether the game was just lost this step.
        ante: Current ante number.
        money: Current money.
        hands_remaining: Hands remaining in the current blind.
        discards_remaining: Discards remaining in the current blind.
    """
    phase: str
    action_valid: bool
    prev_score: int
    new_score: int
    score_target: int
    prev_blinds_beaten: int
    new_blinds_beaten: int
    total_blinds: int
    blind_just_beaten: bool
    won: bool
    lost: bool
    ante: int
    money: int
    hands_remaining: int
    discards_remaining: int


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions.

    Any callable ``(RewardContext) -> float`` satisfies this protocol.
    """

    def __call__(self, ctx: RewardContext) -> float: ...


class DefaultReward:
    """Shaped reward function (the original default).

    Signals:

    * **Invalid action**: ``invalid_action_penalty`` (default -0.01).
    * **Score progress**: Fractional progress toward the blind target,
      scaled by ``score_progress_scale`` (default 0.01), clipped to
      ``[0, score_progress_scale]``.
    * **Blind beaten**: ``blind_beaten_bonus`` (default 1.0) plus a
      progress bonus proportional to ``blinds_beaten / total_blinds``.
    * **Game won**: ``win_reward`` (default 10.0).
    * **Game lost**: ``lose_penalty`` (default -1.0).

    All coefficients can be overridden at construction::

        # Heavier win/lose signal
        reward_fn = DefaultReward(win_reward=50.0, lose_penalty=-5.0)
    """

    def __init__(
        self,
        *,
        invalid_action_penalty: float = -0.01,
        score_progress_scale: float = 0.01,
        blind_beaten_bonus: float = 1.0,
        win_reward: float = 10.0,
        lose_penalty: float = -1.0,
    ):
        self.invalid_action_penalty = invalid_action_penalty
        self.score_progress_scale = score_progress_scale
        self.blind_beaten_bonus = blind_beaten_bonus
        self.win_reward = win_reward
        self.lose_penalty = lose_penalty

    def __call__(self, ctx: RewardContext) -> float:
        reward = 0.0

        # Invalid action penalty
        if not ctx.action_valid:
            reward += self.invalid_action_penalty

        # Score progress shaping
        if ctx.score_target > 0 and ctx.phase == "play":
            score_delta = (ctx.new_score - ctx.prev_score) / ctx.score_target
            reward += min(max(score_delta * self.score_progress_scale, 0.0),
                         self.score_progress_scale)

        # Blind beaten bonus
        if ctx.blind_just_beaten:
            progress_bonus = ctx.new_blinds_beaten / ctx.total_blinds
            reward += self.blind_beaten_bonus + progress_bonus

        # Terminal
        if ctx.won:
            reward += self.win_reward
        if ctx.lost:
            reward += self.lose_penalty

        return reward


class SparseReward:
    """Sparse reward: only signals at game end.

    * ``win_reward`` (default +1.0) when the game is won.
    * ``lose_penalty`` (default -1.0) when the game is lost.
    * 0.0 for all other steps.

    Useful as an ablation baseline to measure whether reward shaping helps.
    """

    def __init__(
        self,
        *,
        win_reward: float = 1.0,
        lose_penalty: float = -1.0,
    ):
        self.win_reward = win_reward
        self.lose_penalty = lose_penalty

    def __call__(self, ctx: RewardContext) -> float:
        if ctx.won:
            return self.win_reward
        if ctx.lost:
            return self.lose_penalty
        return 0.0
