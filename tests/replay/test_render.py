from __future__ import annotations

from pathlib import Path

import pytest

from balatro_gym.replay import Recorder, Replay, render_replay, render_state
from balatro_gym.replay.__main__ import main
from balatro_gym.sim import Game, macros
from balatro_gym.sim import actions as act
from balatro_gym.sim.state import HandType
from scenarios import set_hand, started_round


@pytest.fixture
def replay_file(tmp_path: Path) -> Path:
    game = Game.new("render")
    for level in game.state.hands.values():
        level.mult = 1000  # any hand beats the blind; part of the recorded initial state
    recorder = Recorder(game, game_id="render-1")
    recorder.step(act.SelectBlind())
    recorder.step(act.SortBySuit())
    recorder.step(act.MoveCard(0, 1))
    for action in macros.discard_cards(game.state, [2]):
        recorder.step(action)
    for action in macros.play_cards(game.state, [0, 1]):
        recorder.step(action, annotation={"value": 0.5})
    recorder.step(act.CashOut())
    path = tmp_path / "render.json.gz"
    recorder.replay().save(path)
    return path


def test_state_views() -> None:
    game = Game.new("view")
    text = render_state(game.state)
    assert "Ante 1 · Round 0 · blind select" in text
    assert "> Small Blind (select)" in text
    game.step(act.SelectBlind())
    game.step(act.ToggleCard(0))
    text = render_state(game.state)
    assert "Small Blind · score 0 / 300" in text
    assert "[" in text


def test_round_eval_and_game_over_views() -> None:
    game = started_round("views")
    game.state.hands[HandType.HIGH_CARD].mult = 10**6
    set_hand(game.state, "AS")
    for action in macros.play_cards(game.state, [0]):
        game.step(action)
    assert "Cash out  blind $3 + hands $3 = $6" in render_state(game.state)
    game = started_round("views")
    for _ in range(4):
        for action in macros.play_cards(game.state, [7]):
            game.step(action)
    assert "run over: lost" in render_state(game.state)


def test_replay_log(replay_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["log", str(replay_file)]) == 0
    out = capsys.readouterr().out
    for text in (
        "Replay render-1",
        "#0 select_blind()",
        "Selected the Small Blind: score 300 to win",
        "Drew ",
        "Sorted the hand by suit",
        "Moved ",
        "Discarded ",
        "Played ",
        " chips x 1,000 mult = ",
        "Beat the Small Blind",
        'note: {"value": 0.5}',
        "from cash out",
    ):
        assert text in out


def test_replay_log_detail(replay_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["log", str(replay_file), "--states", "none", "--start", "1", "--stop", "2"])
    out = capsys.readouterr().out
    assert "#1 sort_by_suit()" in out
    assert "#2" not in out
    assert "  | " not in out


def test_show_and_verify(replay_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show", str(replay_file), "--step", "1"]) == 0
    assert "Step 1 of" in capsys.readouterr().out
    assert main(["verify", str(replay_file)]) == 0
    assert capsys.readouterr().out.startswith("ok:")


def test_verify_reports_drift(tmp_path: Path, replay_file: Path) -> None:
    replay = Replay.load(replay_file)
    last = max(replay.keyframes)
    replay.keyframes[last]["state"]["money"] = -1
    path = tmp_path / "drift.json"
    replay.save(path)
    assert main(["verify", str(path)]) == 1


def test_full_states(replay_file: Path) -> None:
    replay = Replay.load(replay_file)
    text = render_replay(replay, states="all")
    assert text.count("  | Ante") == replay.num_steps + 1
