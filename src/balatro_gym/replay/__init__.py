"""Replay files: recording, loading and re-simulating games.

Typical use::

    from balatro_gym.replay import Recorder, Replay, render_replay

    recorder = Recorder(Game.new(seed=7), game_id="run-7")
    ...  # recorder.step(action) for every action
    recorder.replay().save("run-7.json.gz")

    replay = Replay.load("run-7.json.gz")
    replay.verify()
    print(render_replay(replay))
    game = replay.branch(40)  # play on from step 40

From the command line: ``python -m balatro_gym.replay --help``.
"""

from .format import FORMAT_VERSION, Replay, ReplayDriftError, ReplayHeader, Step
from .recorder import KeyframePolicy, Recorder
from .render import render_events, render_replay, render_state, render_step

__all__ = [
    "FORMAT_VERSION",
    "KeyframePolicy",
    "Recorder",
    "Replay",
    "ReplayDriftError",
    "ReplayHeader",
    "Step",
    "render_events",
    "render_replay",
    "render_state",
    "render_step",
]
