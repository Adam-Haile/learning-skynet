"""
Play or watch Hidden Hand games.

    python -m hidden_hand.play greedy tell_reader --vis             # watch two bots in a window
    python -m hidden_hand.play human tell_reader --vis              # play a bot yourself
    python -m hidden_hand.play my_pkg.agents:load_skynet greedy --vis
    python -m hidden_hand.play human human --vis                    # hot seat
    python -m hidden_hand.play greedy tell_reader --games 100       # quick match, text only

Each of the two players is a bot name (random, greedy, tell_reader), ``human``,
or a zero-argument factory ``module:function`` returning an agent with
``act(obs, legal_actions)`` or ``act_batch`` (the quality-check interface).
The first player sits in seat 0 (P0).

``--vis`` opens a window (tkinter) that draws the table after every move, with
pause / step / speed controls. Spectators see every card, who knows it and
whether each tell was true; a human sees only their own view (``--view``
overrides this). Without ``--vis`` each move is one line in the terminal.
"""
from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path
from typing import Any, Callable

from hidden_hand.bots import BOT_REGISTRY, make_bot
from hidden_hand.env import CALL_ACTION, HAND_SIZE, KEEP_ACTION, PEEK_ACTION, TRADE_BASE, HiddenHandEnv, action_name
from hidden_hand.vis import describe_plain


class QuitGame(Exception):
    pass


def load_factory(spec: str) -> Callable[[], Any]:
    """``module:function`` -> the function, importable from the current directory."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"Expected module:function, got {spec!r}")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module_name), attr)


# ----------------------------------------------------------------------
# Players
# ----------------------------------------------------------------------
class Player:
    is_human = False

    def __init__(self, name: str) -> None:
        self.name = name

    def choose(self, obs: dict, legal: list[int]) -> int:
        raise NotImplementedError


class BotPlayer(Player):
    def __init__(self, name: str, seed: int) -> None:
        super().__init__(name)
        self.bot = make_bot(name, seed=seed)

    def choose(self, obs: dict, legal: list[int]) -> int:
        return self.bot.select_action(obs, legal)


class AgentPlayer(Player):
    def __init__(self, spec: str) -> None:
        self.agent = load_factory(spec)()
        super().__init__(str(getattr(self.agent, "name", spec)))

    def choose(self, obs: dict, legal: list[int]) -> int:
        if hasattr(self.agent, "act"):
            return int(self.agent.act(obs, legal))
        return int(self.agent.act_batch([obs], [legal])[0])


class HumanPlayer(Player):
    is_human = True


def make_player(spec: str, seed: int) -> Player:
    if spec == "human":
        return HumanPlayer("human")
    if spec in BOT_REGISTRY:
        return BotPlayer(spec, seed)
    if ":" in spec:
        return AgentPlayer(spec)
    raise SystemExit(f"Unknown player {spec!r}: use human, {', '.join(sorted(BOT_REGISTRY))} or module:factory")


# ----------------------------------------------------------------------
# Text mode (no --vis)
# ----------------------------------------------------------------------
def human_options(env: HiddenHandEnv, legal: list[int]) -> list[str]:
    """The legal moves for the player to move, spelled out with the keys that pick them."""
    me = env.current_player
    hand = env.state.hands[me]
    worst = env.worst_slot(me)
    options = []
    if KEEP_ACTION in legal:
        options.append(f"[k] keep the {env.state.drawn} in slot {worst}")
    if PEEK_ACTION in legal:
        options.append(f"[p] peek at slot {next(i for i in range(HAND_SIZE) if not hand.known_by[i][me])}")
    trades = [a - TRADE_BASE for a in legal if TRADE_BASE <= a < CALL_ACTION]
    if trades:
        options.append(f"[{trades[0]}-{trades[-1]}] trade slot {worst} for P{1 - me}'s card at that slot")
    if CALL_ACTION in legal:
        options.append("[c] call")
    options.append("[q] quit")
    return options


def parse_move(text: str, legal: list[int]) -> int | None:
    """Keys typed by a human -> action id, or None if it is not a legal move."""
    text = text.strip().lower().replace(" ", "").replace("trade", "t")
    if text in ("q", "quit", "exit"):
        raise QuitGame
    named = {"k": KEEP_ACTION, "keep": KEEP_ACTION, "p": PEEK_ACTION, "peek": PEEK_ACTION,
             "c": CALL_ACTION, "call": CALL_ACTION}
    if text in named:
        action = named[text]
    elif text.lstrip("t").isdigit() and int(text.lstrip("t")) < HAND_SIZE:
        action = TRADE_BASE + int(text.lstrip("t"))
    else:
        return None
    return action if action in legal else None


def ask_human(env: HiddenHandEnv, names: list[str]) -> int:
    seat = env.current_player
    print(env.render(observer=seat))
    for action in env.state.history[-6:]:
        print("    " + describe_plain(action, names))
    print(f"  P{seat}, your move. You drew: {env.state.drawn}")
    legal = env.legal_actions()
    print("\n".join("    " + option for option in human_options(env, legal)))
    while True:
        try:
            action = parse_move(input("  > "), legal)
        except EOFError:
            raise QuitGame from None
        if action is not None:
            return action
        print("  Not a legal move here; pick one of the keys above.")


def play_text_game(players: list[Player], seed: int, title: str) -> tuple[HiddenHandEnv, int]:
    """One game, one line per move. Returns the finished env and the number of illegal agent moves."""
    names = [p.name for p in players]
    env = HiddenHandEnv(seed=seed)
    illegal = 0
    print(f"--- {title}: P0 {names[0]} vs P1 {names[1]} (P{env.current_player} starts)")
    while not env.game_over:
        seat = env.current_player
        legal = env.legal_actions()
        note = ""
        if players[seat].is_human:
            action = ask_human(env, names)
        else:
            action = players[seat].choose(env.observe(seat), legal)
            if action not in legal:
                illegal += 1
                note = f"  [illegal {action_name(action)} replaced at random]"
                action = random.Random(seed + env.state.step).choice(legal)
        env.step(action)
        print(f"  step {env.state.step:>2}  {action_name(action):<8}  {describe_plain(env.state.history[-1])}{note}")
    print(env.render())
    print(f"  scores {env.scores[0]} : {env.scores[1]}  ->  {result_text(env, names)}")
    return env, illegal


def result_text(env: HiddenHandEnv, names: list[str]) -> str:
    return "draw" if env.winner == -1 else f"P{env.winner} {names[env.winner]} wins"


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def run_window(players: list[Player], seeds: list[int], delay: float, view: str, paused: bool) -> tuple[list[int], int]:
    try:
        import tkinter

        from hidden_hand.window import GameWindow
    except ImportError:
        raise SystemExit("--vis needs tkinter, which this Python was built without. Install it "
                         "(e.g. `brew install python-tk` or `apt install python3-tk`) or run without --vis.") from None
    names = [p.name for p in players]

    def log(index: int, env: HiddenHandEnv) -> None:
        print(f"  game {index + 1}/{len(seeds)} (seed {seeds[index]}): "
              f"{env.scores[0]} : {env.scores[1]}  ->  {result_text(env, names)}", flush=True)

    try:
        window = GameWindow(players, seeds, delay=delay, view_mode=view, paused=paused, on_game_over=log)
    except tkinter.TclError as exc:
        raise SystemExit(f"Could not open a window ({exc}). Run without --vis on a machine with no display.") from None
    return window.run(), window.illegal


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("p0", help="seat 0: human, random, greedy, tell_reader or module:factory")
    parser.add_argument("p1", help="seat 1: same choices")
    parser.add_argument("--vis", action="store_true", help="watch (or play) in a window")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="game g uses seed + g")
    parser.add_argument("--delay", type=float, default=1.0, help="--vis: seconds between bot moves (adjustable live)")
    parser.add_argument("--paused", action="store_true", help="--vis: start paused and advance with Step / Right arrow")
    parser.add_argument("--view", choices=("auto", "all", "0", "1"), default="auto",
                        help="--vis: whose cards to show: auto = the human's seat if one human plays, else everything")
    args = parser.parse_args(argv)

    players = [make_player(args.p0, args.seed), make_player(args.p1, args.seed + 1)]
    names = [p.name for p in players]
    seeds = [args.seed + g for g in range(args.games)]

    if args.vis:
        tally, illegal = run_window(players, seeds, args.delay, args.view, args.paused)
        played = sum(tally)
    else:
        tally, illegal, played = [0, 0, 0], 0, 0
        try:
            for g, seed in enumerate(seeds):
                env, bad = play_text_game(players, seed, f"game {g + 1}/{args.games} · seed {seed}")
                tally[env.winner] += 1
                illegal += bad
                played += 1
        except (QuitGame, KeyboardInterrupt):
            print("\n  stopped.")
    print(f"\n  {played} game(s): P0 {names[0]} {tally[0]} · P1 {names[1]} {tally[1]} · draws {tally[2]}"
          + (f" · {illegal} illegal agent actions" if illegal else ""))


if __name__ == "__main__":
    main()
