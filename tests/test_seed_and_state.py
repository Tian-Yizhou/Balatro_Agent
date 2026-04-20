"""Tests for episode seed IDs, state serialization, and resume functionality."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import balatro_gym  # noqa: F401
from balatro_gym.core.seed_id import (
    generate_seed_id,
    parse_seed_id,
    seed_id_to_game_seed,
    _encode_base36,
    _decode_base36,
)
from balatro_gym.core.card import Card, Rank, Suit, Enhancement, Edition, Seal
from balatro_gym.core.game_state import GameState
from balatro_gym.core.joker import create_joker
from balatro_gym.envs.balatro_env import BalatroEnv
from balatro_gym.envs.configs import GameConfig
from balatro_gym.wrappers import RolloutRecorder


# -----------------------------------------------------------------------
# Seed ID
# -----------------------------------------------------------------------


class TestBase36:
    def test_encode_zero(self):
        assert _encode_base36(0) == "00000000"

    def test_encode_small(self):
        assert _encode_base36(42) == "00000016"

    def test_decode_roundtrip(self):
        for n in [0, 1, 42, 999, 123456, 2**31 - 1]:
            assert _decode_base36(_encode_base36(n)) == n

    def test_encode_max(self):
        # 36^8 - 1
        s = _encode_base36(36**8 - 1)
        assert s == "ZZZZZZZZ"
        assert _decode_base36(s) == 36**8 - 1

    def test_encode_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _encode_base36(-1)

    def test_encode_overflow_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            _encode_base36(36**8)


class TestSeedIdGeneration:
    def test_format(self):
        ts = datetime(2026, 4, 18, 14, 30, tzinfo=timezone.utc)
        sid = generate_seed_id(42, timestamp=ts)
        assert sid == "20260418-1430-00000016"

    def test_format_structure(self):
        sid = generate_seed_id(0)
        parts = sid.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 4  # HHMM
        assert len(parts[2]) == 8  # base36

    def test_different_seeds_different_ids(self):
        ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        id1 = generate_seed_id(100, timestamp=ts)
        id2 = generate_seed_id(200, timestamp=ts)
        assert id1 != id2

    def test_same_seed_different_timestamps(self):
        ts1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
        id1 = generate_seed_id(42, timestamp=ts1)
        id2 = generate_seed_id(42, timestamp=ts2)
        assert id1 != id2

    def test_large_seed(self):
        sid = generate_seed_id(2**31 - 1)
        parsed = parse_seed_id(sid)
        assert parsed["game_seed"] == 2**31 - 1


class TestSeedIdParsing:
    def test_roundtrip(self):
        ts = datetime(2026, 4, 18, 14, 30, tzinfo=timezone.utc)
        sid = generate_seed_id(42, timestamp=ts)
        parsed = parse_seed_id(sid)
        assert parsed["game_seed"] == 42
        assert parsed["timestamp"] == "20260418-1430"
        assert parsed["seed_str"] == "00000016"

    def test_extract_game_seed(self):
        sid = generate_seed_id(999)
        assert seed_id_to_game_seed(sid) == 999

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            parse_seed_id("not-a-valid-seed")

    def test_wrong_lengths_raise(self):
        with pytest.raises(ValueError, match="Invalid"):
            parse_seed_id("2026041-1430-00000016")  # date too short


# -----------------------------------------------------------------------
# Card serialization
# -----------------------------------------------------------------------


class TestCardSerialization:
    def test_basic_roundtrip(self):
        card = Card(Rank.ACE, Suit.SPADES)
        d = card.to_dict()
        restored = Card.from_dict(d)
        assert restored.rank == Rank.ACE
        assert restored.suit == Suit.SPADES
        assert restored.uid == card.uid

    def test_with_properties(self):
        card = Card(Rank.TEN, Suit.HEARTS,
                    enhancement=Enhancement.GLASS,
                    edition=Edition.FOIL,
                    seal=Seal.RED,
                    face_down=True)
        d = card.to_dict()
        restored = Card.from_dict(d)
        assert restored.enhancement == Enhancement.GLASS
        assert restored.edition == Edition.FOIL
        assert restored.seal == Seal.RED
        assert restored.face_down is True

    def test_dict_is_json_serializable(self):
        card = Card(Rank.KING, Suit.DIAMONDS,
                    enhancement=Enhancement.WILD,
                    edition=Edition.POLYCHROME,
                    seal=Seal.GOLD)
        d = card.to_dict()
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)
        restored = Card.from_dict(restored_dict)
        assert restored.rank == Rank.KING
        assert restored.enhancement == Enhancement.WILD

    def test_none_properties(self):
        card = Card(Rank.TWO, Suit.CLUBS)
        d = card.to_dict()
        assert d["enhancement"] is None
        assert d["edition"] is None
        assert d["seal"] is None
        restored = Card.from_dict(d)
        assert restored.enhancement is None


# -----------------------------------------------------------------------
# Joker state serialization
# -----------------------------------------------------------------------


class TestJokerStateSerialization:
    def test_stateless_joker(self):
        joker = create_joker("joker_basic")
        state = joker.get_state()
        assert isinstance(state, dict)

    def test_stateful_joker_roundtrip(self):
        joker = create_joker("ice_cream")
        # Simulate some hands played
        joker._internal_state["chips"] = 75
        state = joker.get_state()
        assert state["chips"] == 75

        # Restore into fresh joker
        fresh = create_joker("ice_cream")
        assert fresh._internal_state["chips"] == 100  # default
        fresh.set_state(state)
        assert fresh._internal_state["chips"] == 75

    def test_runner_state(self):
        joker = create_joker("runner")
        joker._internal_state["chips"] = 45
        state = joker.get_state()

        fresh = create_joker("runner")
        fresh.set_state(state)
        assert fresh._internal_state["chips"] == 45


# -----------------------------------------------------------------------
# GameState serialization
# -----------------------------------------------------------------------


class TestGameStateSerialization:
    def _make_game(self, seed=42):
        gs = GameState(
            num_antes=4,
            hands_per_round=4,
            discards_per_round=3,
            max_jokers=5,
            starting_money=6,
            available_joker_ids=["joker_basic", "greedy_joker", "ice_cream"],
            starting_joker_ids=["joker_basic"],
            available_consumable_ids=["c_pluto", "c_mercury"],
            consumable_slots=2,
            seed=seed,
        )
        gs.reset()
        return gs

    def test_serialize_returns_dict(self):
        gs = self._make_game()
        data = gs.serialize()
        assert isinstance(data, dict)
        assert "config" in data
        assert "hand" in data
        assert "rng_state" in data

    def test_roundtrip_preserves_scalars(self):
        gs = self._make_game()
        # Play a hand to change state
        gs.play_hand([0, 1])
        data = gs.serialize()
        restored = GameState.deserialize(data)

        assert restored.money == gs.money
        assert restored.ante == gs.ante
        assert restored.blind_index == gs.blind_index
        assert restored.current_score == gs.current_score
        assert restored.hands_remaining == gs.hands_remaining
        assert restored.phase == gs.phase
        assert restored.total_hands_played == gs.total_hands_played
        assert restored.blinds_beaten == gs.blinds_beaten

    def test_roundtrip_preserves_hand(self):
        gs = self._make_game()
        data = gs.serialize()
        restored = GameState.deserialize(data)

        assert len(restored.hand) == len(gs.hand)
        for orig, rest in zip(gs.hand, restored.hand):
            assert rest.rank == orig.rank
            assert rest.suit == orig.suit
            assert rest.uid == orig.uid

    def test_roundtrip_preserves_jokers(self):
        gs = self._make_game()
        data = gs.serialize()
        restored = GameState.deserialize(data)

        assert len(restored.jokers) == len(gs.jokers)
        for orig, rest in zip(gs.jokers, restored.jokers):
            assert rest.INFO.id == orig.INFO.id

    def test_roundtrip_preserves_rng(self):
        """After deserialization, the RNG should produce the same sequence."""
        gs = self._make_game()
        data = gs.serialize()
        restored = GameState.deserialize(data)

        # Both should produce the same next random number
        orig_val = gs.rng.random()
        rest_val = restored.rng.random()
        assert orig_val == rest_val

    def test_restored_game_continues_identically(self):
        """Play from the same checkpoint should produce identical results."""
        gs = self._make_game(seed=123)
        # Play a few hands
        gs.play_hand([0, 1])

        # Save checkpoint
        data = gs.serialize()
        gs_copy = GameState.deserialize(data)

        # Play the same action on both
        score_orig, _ = gs.play_hand([0, 1, 2])
        score_copy, _ = gs_copy.play_hand([0, 1, 2])

        assert score_orig == score_copy
        assert gs.current_score == gs_copy.current_score
        assert gs.money == gs_copy.money


# -----------------------------------------------------------------------
# Env-level save/load
# -----------------------------------------------------------------------


class TestEnvSaveLoad:
    def test_save_returns_dict(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        env.reset(seed=42)
        checkpoint = env.save_state()
        assert isinstance(checkpoint, dict)
        assert "game_state" in checkpoint
        assert "episode_seed_id" in checkpoint

    def test_load_restores_obs(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs1, info1 = env.reset(seed=42)

        # Take a few actions
        for _ in range(5):
            mask = info1["action_mask"]
            valid = np.where(mask)[0]
            obs1, _, term, _, info1 = env.step(int(valid[0]))
            if term:
                break

        # Save
        checkpoint = env.save_state()

        # Create fresh env and load
        env2 = BalatroEnv(config=GameConfig.easy(seed=99))
        env2.reset(seed=99)  # different initial state
        obs2, info2 = env2.load_state(checkpoint)

        np.testing.assert_array_equal(obs1, obs2)
        assert info1["phase"] == info2["phase"]
        assert info1["score"] == info2["score"]
        assert info1["money"] == info2["money"]

    def test_loaded_env_continues_identically(self):
        """After loading, env should produce same results as original."""
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, info = env.reset(seed=42)

        # Play a few steps
        rng = np.random.default_rng(0)
        for _ in range(10):
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            obs, _, term, _, info = env.step(int(valid[0]))
            if term:
                break

        # Save checkpoint
        checkpoint = env.save_state()

        # Continue original
        mask = info["action_mask"]
        valid = np.where(mask)[0]
        if len(valid) > 0:
            obs_orig, rew_orig, _, _, _ = env.step(int(valid[0]))

            # Load into fresh env and take same action
            env2 = BalatroEnv(config=GameConfig.easy())
            env2.reset(seed=0)
            env2.load_state(checkpoint)
            obs_copy, rew_copy, _, _, _ = env2.step(int(valid[0]))

            np.testing.assert_array_equal(obs_orig, obs_copy)
            assert rew_orig == rew_copy


# -----------------------------------------------------------------------
# from_seed_id
# -----------------------------------------------------------------------


class TestFromSeedId:
    def test_creates_same_game(self):
        """Two envs created with the same seed ID should start identically."""
        env1 = BalatroEnv(config=GameConfig.easy())
        obs1, info1 = env1.reset(seed=42)
        seed_id = info1["episode_seed_id"]

        env2, obs2, info2 = BalatroEnv.from_seed_id(
            seed_id, config=GameConfig.easy()
        )
        np.testing.assert_array_equal(obs1, obs2)

    def test_preserves_seed_id(self):
        ts = datetime(2026, 4, 18, 14, 30, tzinfo=timezone.utc)
        sid = generate_seed_id(42, timestamp=ts)

        env, obs, info = BalatroEnv.from_seed_id(sid, config=GameConfig.easy())
        assert env.episode_seed_id == sid
        assert info["episode_seed_id"] == sid


# -----------------------------------------------------------------------
# Seed ID in wrappers
# -----------------------------------------------------------------------


class TestSeedIdInWrappers:
    def test_rollout_stores_seed_id(self, tmp_path):
        env = RolloutRecorder(
            gym.make("Balatro-Easy-v0"), save_dir=tmp_path / "rollouts"
        )
        obs, info = env.reset(seed=42)
        seed_id = info["episode_seed_id"]

        # Play to completion
        for _ in range(2000):
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            obs, _, term, _, info = env.step(int(valid[0]))
            if term:
                break

        # Load and check seed_id is stored
        npz_files = list((tmp_path / "rollouts").glob("*.npz"))
        assert len(npz_files) == 1
        data = RolloutRecorder.load(npz_files[0])
        assert "episode_seed_id" in data
        assert data["episode_seed_id"] == seed_id

    def test_seed_id_in_info_dict(self):
        env = BalatroEnv(config=GameConfig.easy(seed=42))
        obs, info = env.reset(seed=42)
        assert "episode_seed_id" in info
        sid = info["episode_seed_id"]
        assert len(sid) == 22  # YYYYMMDD-HHMM-XXXXXXXX
        parsed = parse_seed_id(sid)
        assert parsed["game_seed"] == 42
