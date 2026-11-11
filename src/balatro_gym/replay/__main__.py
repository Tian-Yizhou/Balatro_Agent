"""Command-line viewer for replay files.

Examples::

    python -m balatro_gym.replay log run.json.gz
    python -m balatro_gym.replay log run.json.gz --start 40 --stop 60 --states all
    python -m balatro_gym.replay show run.json.gz --step 40
    python -m balatro_gym.replay verify run.json.gz
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from balatro_gym.sim import RULES_VERSION

from .format import Replay, ReplayDriftError
from .render import render_replay, render_state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m balatro_gym.replay", description="Inspect a recorded game."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    log = commands.add_parser("log", help="print the game step by step")
    log.add_argument("file")
    log.add_argument("--start", type=int, default=0, help="first step to print")
    log.add_argument("--stop", type=int, default=None, help="step to stop before")
    log.add_argument(
        "--states",
        choices=("all", "phases", "none"),
        default="phases",
        help="when to print the full state (default: whenever the phase changes)",
    )

    show = commands.add_parser("show", help="print the state at one step")
    show.add_argument("file")
    show.add_argument("--step", type=int, default=None, help="default: the final step")

    verify = commands.add_parser("verify", help="re-simulate and check every keyframe")
    verify.add_argument("file")

    args = parser.parse_args(argv)
    replay = Replay.load(args.file)

    if args.command == "log":
        print(render_replay(replay, args.start, args.stop, states=args.states))
    elif args.command == "show":
        index = replay.num_steps if args.step is None else args.step
        print(f"Step {index} of {replay.num_steps}")
        print(render_state(replay.state_at(index)))
    else:
        recorded = replay.header.rules_version
        try:
            replay.verify()
        except ReplayDriftError as exc:
            print(f"drift: {exc} (recorded with rules v{recorded}, engine v{RULES_VERSION})")
            return 1
        print(f"ok: {replay.num_steps} steps, {len(replay.keyframes)} keyframes reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
