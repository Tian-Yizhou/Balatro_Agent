"""User-facing action journeys from an agent decision through ``env.step``.

These tests cover the actions exposed by :class:`BalatroEnv`: play, discard,
buy, sell, reroll, and skip.  Tests interact through the Gym API and inspect
the public ``info`` payload/action mask rather than calling ``GameState``
actions directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.baselines.heuristic_agent import HeuristicAgent
from agent.baselines.random_agent import RandomAgent
from agent.llm.renderer import GameStateRenderer
from balatro_gym.core.card import Card, Rank, Suit
from balatro_gym.envs.balatro_env import (
    BUY_OFFSET,
    CARD_SUBSETS,
    DISCARD_OFFSET,
    PLAY_OFFSET,
    REROLL_ACTION,
    SELL_OFFSET,
    SKIP_ACTION,
    BalatroEnv,
)
from balatro_gym.envs.configs import GameConfig


def _subset_action(offset: int, selected: tuple[int, ...]) -> int:
    return offset + CARD_SUBSETS.index(selected)


def _play_journey_env(seed: int = 7) -> tuple[BalatroEnv, np.ndarray, dict]:
    env = BalatroEnv(config=GameConfig.medium(seed=seed))
    obs, info = env.reset(seed=seed)
    return env, obs, info


def _shop_journey_env() -> tuple[BalatroEnv, np.ndarray, dict]:
    """Reach the shop in one legal user action with ample shop funds."""
    config = GameConfig(
        num_antes=2,
        hands_per_round=2,
        discards_per_round=2,
        hand_size=8,
        max_jokers=25,
        starting_money=100,
        shop_slots=2,
        reroll_base_cost=5,
        consumable_slots=2,
        joker_pool=["joker_basic"],
        starting_joker_ids=["joker_basic"] * 20,
        consumable_pool=["c_mercury"],
        seed=17,
    )
    env = BalatroEnv(config=config)
    _, info = env.reset(seed=17)
    play_one = _subset_action(PLAY_OFFSET, (0,))
    assert info["action_mask"][play_one]

    obs, _, terminated, truncated, info = env.step(play_one)

    assert not terminated
    assert not truncated
    assert info["phase"] == "shop"
    return env, obs, info


class TestUserPlayPhaseActions:
    def test_user_play_action_scores_and_spends_a_hand(self):
        env = BalatroEnv(config=GameConfig.medium(seed=7))
        _, info = env.reset(seed=7)
        play_one = _subset_action(PLAY_OFFSET, (0,))

        assert info["action_mask"][play_one]
        previous_hands = info["hands_remaining"]

        _, reward, terminated, truncated, info = env.step(play_one)

        assert not terminated
        assert not truncated
        assert info["score"] > 0
        assert info["hands_remaining"] == previous_hands - 1
        assert isinstance(reward, float)

    def test_user_discard_action_refreshes_hand_without_scoring(self):
        env = BalatroEnv(config=GameConfig.medium(seed=9))
        _, info = env.reset(seed=9)
        discard_two = _subset_action(DISCARD_OFFSET, (0, 1))

        assert info["action_mask"][discard_two]
        previous_discards = info["discards_remaining"]
        previous_hands = info["hands_remaining"]

        _, reward, terminated, truncated, info = env.step(discard_two)

        assert not terminated
        assert not truncated
        assert reward == 0.0
        assert info["score"] == 0
        assert info["hands_remaining"] == previous_hands
        assert info["discards_remaining"] == previous_discards - 1

    def test_shop_action_during_play_is_rejected_without_state_change(self):
        env, _, info = _play_journey_env()
        before = (info["score"], info["money"], info["hands_remaining"])

        assert not info["action_mask"][SKIP_ACTION]

        _, reward, _, _, info = env.step(SKIP_ACTION)

        assert reward == pytest.approx(-0.01)
        assert (info["score"], info["money"], info["hands_remaining"]) == before

    def test_discard_after_discards_exhausted_is_rejected(self):
        env, _, _ = _play_journey_env()
        env._game.discards_remaining = 0
        info = env._build_info()
        action = _subset_action(DISCARD_OFFSET, (0,))

        assert not info["action_mask"][action]

        _, reward, _, _, info = env.step(action)

        assert reward == pytest.approx(-0.01)
        assert info["discards_remaining"] == 0

    @pytest.mark.parametrize("subset_index", range(len(CARD_SUBSETS)))
    def test_every_play_card_selection_action_is_accepted(self, subset_index):
        env, _, info = _play_journey_env(seed=subset_index)
        action = PLAY_OFFSET + subset_index

        assert info["action_mask"][action]

        env.step(action)

        assert env._game.total_hands_played == 1

    @pytest.mark.parametrize("subset_index", range(len(CARD_SUBSETS)))
    def test_every_discard_card_selection_action_is_accepted(self, subset_index):
        env, _, info = _play_journey_env(seed=subset_index)
        action = DISCARD_OFFSET + subset_index
        previous_discards = info["discards_remaining"]

        assert info["action_mask"][action]

        _, _, _, _, info = env.step(action)

        assert info["score"] == 0
        assert info["discards_remaining"] == previous_discards - 1


class TestUserShopActions:
    @pytest.mark.parametrize("slot", (0, 1))
    def test_user_buys_a_joker_from_each_affordable_masked_slot(self, slot):
        env, _, info = _shop_journey_env()
        buy_joker = BUY_OFFSET + slot

        assert info["action_mask"][buy_joker]
        previous_money = info["money"]
        previous_jokers = info["num_jokers"]

        _, _, _, _, info = env.step(buy_joker)

        assert info["money"] < previous_money
        assert info["num_jokers"] == previous_jokers + 1
        assert not info["action_mask"][buy_joker]

    def test_user_buys_a_consumable_through_the_shop_slot_action(self):
        env, _, info = _shop_journey_env()
        buy_consumable = BUY_OFFSET + 2

        assert info["action_mask"][buy_consumable]
        previous_money = info["money"]

        _, _, _, _, info = env.step(buy_consumable)

        assert info["money"] < previous_money
        assert info["num_consumables"] == 1
        assert not info["action_mask"][buy_consumable]

    @pytest.mark.parametrize("joker_index", range(5))
    def test_user_sells_each_exposed_joker_slot_and_receives_money(self, joker_index):
        env, _, info = _shop_journey_env()
        sell_joker = SELL_OFFSET + joker_index

        assert info["action_mask"][sell_joker]
        previous_money = info["money"]
        previous_jokers = info["num_jokers"]

        _, _, _, _, info = env.step(sell_joker)

        assert info["money"] > previous_money
        assert info["num_jokers"] == previous_jokers - 1

    def test_user_rerolls_the_shop_and_pays_the_visible_action_cost(self):
        env, _, info = _shop_journey_env()

        assert info["action_mask"][REROLL_ACTION]
        previous_money = info["money"]

        _, _, _, _, info = env.step(REROLL_ACTION)

        assert info["phase"] == "shop"
        assert info["money"] == previous_money - 5
        assert info["action_mask"][REROLL_ACTION]
        assert info["action_mask"][SKIP_ACTION]

    def test_user_skips_the_shop_and_starts_the_next_blind(self):
        env, _, info = _shop_journey_env()

        assert info["action_mask"][SKIP_ACTION]

        _, _, terminated, truncated, info = env.step(SKIP_ACTION)

        assert not terminated
        assert not truncated
        assert info["phase"] == "play"
        assert info["blind_index"] == 1
        assert info["action_mask"][PLAY_OFFSET]
        assert not info["action_mask"][SKIP_ACTION]

    @pytest.mark.parametrize("invalid_action", (BUY_OFFSET, REROLL_ACTION))
    def test_masked_unaffordable_shop_action_is_penalized(self, invalid_action):
        env, _, info = _shop_journey_env()
        env._game.money = 0
        info = env._build_info()
        before = (info["money"], info["num_jokers"], info["num_consumables"])

        assert not info["action_mask"][invalid_action]

        _, reward, _, _, info = env.step(invalid_action)

        assert reward == pytest.approx(-0.01)
        assert (info["money"], info["num_jokers"], info["num_consumables"]) == before

    def test_masked_sell_empty_joker_slot_is_penalized(self):
        env, _, info = _shop_journey_env()
        invalid_sell = SELL_OFFSET + 4
        env._game.jokers = []
        info = env._build_info()

        assert not info["action_mask"][invalid_sell]

        _, reward, _, _, info = env.step(invalid_sell)

        assert reward == pytest.approx(-0.01)
        assert info["num_jokers"] == 0


class TestAgentToEnvironmentJourneys:
    @pytest.mark.parametrize(
        "action",
        [PLAY_OFFSET + i for i in range(len(CARD_SUBSETS))]
        + [DISCARD_OFFSET + i for i in range(len(CARD_SUBSETS))],
    )
    def test_agent_can_forward_every_play_phase_action_id(self, action):
        agent = RandomAgent(seed=31)
        env, obs, info = _play_journey_env(seed=action)
        user_choice = np.zeros_like(info["action_mask"])
        user_choice[action] = True
        visible_actions = {
            item["action_id"]
            for item in GameStateRenderer(env).render(info)["valid_actions"]
        }

        assert info["action_mask"][action]
        assert action in visible_actions
        selected = agent.act(obs, {**info, "action_mask": user_choice})

        assert selected == action
        env.step(selected)

    def test_random_agent_dispatches_each_user_action_to_the_environment(self):
        agent = RandomAgent(seed=31)

        def dispatch(env: BalatroEnv, obs: np.ndarray, info: dict, action: int) -> dict:
            assert info["action_mask"][action]
            assert action in {
                item["action_id"]
                for item in GameStateRenderer(env).render(info)["valid_actions"]
            }
            user_choices = np.zeros_like(info["action_mask"])
            user_choices[action] = True
            action = agent.act(obs, {**info, "action_mask": user_choices})
            assert user_choices[action]
            return env.step(action)[-1]

        for action in (
            _subset_action(PLAY_OFFSET, (0,)),
            _subset_action(DISCARD_OFFSET, (0,)),
        ):
            env = BalatroEnv(config=GameConfig.medium(seed=31))
            obs, info = env.reset(seed=31)
            dispatch(env, obs, info, action)

        for action in (
            BUY_OFFSET,
            BUY_OFFSET + 1,
            BUY_OFFSET + 2,
            SELL_OFFSET,
            SELL_OFFSET + 1,
            SELL_OFFSET + 2,
            SELL_OFFSET + 3,
            SELL_OFFSET + 4,
            REROLL_ACTION,
            SKIP_ACTION,
        ):
            env, obs, info = _shop_journey_env()
            dispatch(env, obs, info, action)

    def test_random_agent_output_is_accepted_by_play_and_shop_masks(self):
        agent = RandomAgent(seed=12)
        env = BalatroEnv(config=GameConfig.easy(seed=12))
        obs, info = env.reset(seed=12)

        play_action = agent.act(obs, info)
        assert info["action_mask"][play_action]
        env.step(play_action)

        shop_env, shop_obs, shop_info = _shop_journey_env()
        shop_action = agent.act(shop_obs, shop_info)
        assert shop_info["action_mask"][shop_action]
        shop_env.step(shop_action)

    def test_heuristic_agent_buys_an_affordable_item_in_the_shop(self):
        env, obs, info = _shop_journey_env()
        agent = HeuristicAgent(env=env, seed=22)
        previous_jokers = info["num_jokers"]

        action = agent.act(obs, info)

        assert info["action_mask"][action]
        assert action in (BUY_OFFSET, BUY_OFFSET + 1)

        _, _, _, _, info = env.step(action)

        assert info["num_jokers"] == previous_jokers + 1

    def test_heuristic_agent_discards_weak_cards_while_searching_for_a_hand(self):
        env, obs, _ = _play_journey_env()
        env._game.hand = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.FOUR, Suit.SPADES),
            Card(Rank.SIX, Suit.DIAMONDS),
            Card(Rank.EIGHT, Suit.CLUBS),
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.SPADES),
            Card(Rank.KING, Suit.DIAMONDS),
            Card(Rank.ACE, Suit.CLUBS),
        ]
        info = env._build_info()
        agent = HeuristicAgent(env=env, seed=22)

        action = agent.act(obs, info)

        assert DISCARD_OFFSET <= action < BUY_OFFSET
        previous_discards = info["discards_remaining"]
        _, _, _, _, info = env.step(action)
        assert info["discards_remaining"] == previous_discards - 1

    def test_heuristic_agent_plays_when_no_discard_remains(self):
        env, obs, _ = _play_journey_env()
        env._game.discards_remaining = 0
        info = env._build_info()
        agent = HeuristicAgent(env=env, seed=22)

        action = agent.act(obs, info)

        assert PLAY_OFFSET <= action < DISCARD_OFFSET
        env.step(action)
        assert env._game.total_hands_played == 1

    def test_heuristic_agent_skips_shop_when_nothing_is_affordable(self):
        env, obs, info = _shop_journey_env()
        env._game.jokers = []
        env._game.money = 0
        info = env._build_info()
        agent = HeuristicAgent(env=env, seed=22)

        action = agent.act(obs, info)

        assert action == SKIP_ACTION
        _, _, _, _, info = env.step(action)
        assert info["phase"] == "play"


class TestHumanReadableActionBoundary:
    def test_renderer_presents_play_and_discard_choices_a_user_can_execute(self):
        env, _, info = _play_journey_env()
        actions = GameStateRenderer(env).render(info)["valid_actions"]
        by_type = {action["type"]: action for action in actions}

        assert {"play", "discard"} <= set(by_type)

        for action_type in ("play", "discard"):
            new_env, _, new_info = _play_journey_env()
            action_id = {
                action["type"]: action
                for action in GameStateRenderer(new_env).render(new_info)["valid_actions"]
            }[action_type]["action_id"]
            assert new_info["action_mask"][action_id]
            new_env.step(action_id)

    @pytest.mark.parametrize("action_type", ("buy", "sell", "reroll", "skip"))
    def test_renderer_presents_each_shop_choice_a_user_can_execute(self, action_type):
        env, _, info = _shop_journey_env()
        actions = GameStateRenderer(env).render(info)["valid_actions"]
        action = next(action for action in actions if action["type"] == action_type)

        assert info["action_mask"][action["action_id"]]

        env.step(action["action_id"])

    def test_renderer_presents_every_exposed_buy_and_sell_slot(self):
        env, _, info = _shop_journey_env()
        actions = GameStateRenderer(env).render(info)["valid_actions"]
        action_ids = {action["action_id"] for action in actions}

        assert {BUY_OFFSET, BUY_OFFSET + 1, BUY_OFFSET + 2} <= action_ids
        assert {SELL_OFFSET + i for i in range(5)} <= action_ids
