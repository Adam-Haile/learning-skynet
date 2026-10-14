"""
Run the quality checks from the command line.

    python -m hidden_hand.checks \
        --muzero my_package.agents:load_muzero \
        --skynet my_package.agents:load_skynet \
        --muzero-log runs/muzero/history.json \
        --skynet-log runs/skynet/history.json

``--muzero`` / ``--skynet`` name a zero-argument factory ``module:function``
that returns an agent (see README.md for the agent interface).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hidden_hand.checks.harness import run_checks
from hidden_hand.play import load_factory as _load_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--muzero", required=True, help="module:factory returning your MuZero agent")
    parser.add_argument("--skynet", required=True, help="module:factory returning your SkyNet agent")
    parser.add_argument("--muzero-log", default=None, help="training log (JSON list of dicts or CSV)")
    parser.add_argument("--skynet-log", default=None)
    parser.add_argument("--targets", default=None, help="override the bundled targets.json")
    parser.add_argument("--tiers", default="1,2,3,4")
    parser.add_argument("--h2h-games", type=int, default=600)
    parser.add_argument("--bot-games", type=int, default=300)
    parser.add_argument("--probe-games", type=int, default=300)
    parser.add_argument("--ablation-games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--json-out", default=None, help="also write the results as JSON")
    args = parser.parse_args()

    report = run_checks(
        muzero=_load_factory(args.muzero)(),
        skynet=_load_factory(args.skynet)(),
        muzero_log=args.muzero_log,
        skynet_log=args.skynet_log,
        targets=args.targets,
        h2h_games=args.h2h_games,
        bot_games=args.bot_games,
        probe_games=args.probe_games,
        ablation_games=args.ablation_games,
        seed=args.seed,
        tiers=[int(t) for t in args.tiers.split(",")],
    )
    print()
    print(report.table())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.to_json(), indent=2))
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
