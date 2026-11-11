from __future__ import annotations

from collections import Counter

import pytest

from balatro_gym.sim.rng import Rng, stable_hash


def test_stable_hash_never_changes() -> None:
    # Saved games and replays depend on these exact values.
    assert stable_hash("") == 0xB4B2797457A0A6E4
    assert stable_hash("boss") == 0xD2C695FE876CA76A
    assert stable_hash("boss") != stable_hash("boss ")


def test_draws_are_reproducible() -> None:
    a, b = Rng("seed"), Rng("seed")
    assert [a.random("k") for _ in range(10)] == [b.random("k") for _ in range(10)]


def test_draws_never_change() -> None:
    # Changing the generator would silently change every recorded game.
    rng = Rng("ABC123")
    assert [rng.randint("boss", 1, 1000) for _ in range(5)] == [947, 134, 86, 979, 996]
    assert Rng("ABC123").random("nr1") == 0.4901796776660965


def test_streams_are_independent() -> None:
    a, b = Rng("seed"), Rng("seed")
    for _ in range(100):
        a.random("other")
    assert a.random("boss") == b.random("boss")


def test_seeds_differ() -> None:
    assert Rng("a").random("k") != Rng("b").random("k")


def test_counters_track_draws_per_stream() -> None:
    rng = Rng("seed")
    rng.random("x")
    rng.randint("x", 1, 6)
    rng.shuffle("y", list(range(10)))
    assert rng.counters == {"x": 2, "y": 1}


def test_round_trip_resumes_the_same_sequence() -> None:
    rng = Rng("seed")
    for _ in range(5):
        rng.random("k")
    restored = Rng.from_dict(rng.to_dict())
    assert restored == rng
    assert restored.random("k") == rng.random("k")


def test_randint_bounds_and_coverage() -> None:
    rng = Rng("seed")
    counts = Counter(rng.randint("d", 1, 6) for _ in range(6000))
    assert set(counts) == {1, 2, 3, 4, 5, 6}
    assert all(800 < n < 1200 for n in counts.values())


def test_random_is_in_unit_interval() -> None:
    rng = Rng("seed")
    values = [rng.random("u") for _ in range(10000)]
    assert all(0.0 <= v < 1.0 for v in values)
    assert 0.48 < sum(values) / len(values) < 0.52


def test_shuffle_is_a_permutation_and_uniform() -> None:
    rng = Rng("seed")
    firsts: Counter[int] = Counter()
    for _ in range(4000):
        items = list(range(4))
        rng.shuffle("s", items)
        assert sorted(items) == [0, 1, 2, 3]
        firsts[items[0]] += 1
    assert all(850 < n < 1150 for n in firsts.values())


def test_invalid_arguments() -> None:
    rng = Rng("seed")
    with pytest.raises(ValueError, match="empty range"):
        rng.randint("k", 3, 2)
    with pytest.raises(ValueError, match="empty sequence"):
        rng.choice("k", [])
