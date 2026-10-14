"""
Simple opponents for Hidden Hand.

Every bot exposes ``select_action(obs, legal_actions) -> int`` like the SkyNet
repo's heuristic bots, so any of them can sit in either seat.

* ``RandomBot``     uniform over legal actions.
* ``GreedyBot``     manages its own hand only: keep draws that beat its worst
                    card, peek unknown cards, call once its hand is known and
                    low. It never trades and ignores the opponent entirely.
* ``TellReaderBot`` keeps a per-card estimate of both hands from the public
                    tells (following cards through trades), trades its worst
                    card for the opponent card that looks lowest, and calls
                    when it believes it is ahead.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from hidden_hand.env import (
    CALL_ACTION,
    HAND_SIZE,
    KEEP_ACTION,
    PEEK_ACTION,
    TRADE_BASE,
    UNKNOWN_VALUE,
)

PRIOR_MEAN = 4.5
# Posterior means for a single card given its latest tell, with the default
# deck (uniform 0..9, HIGH means >= 5) and 75% tell accuracy.
MEAN_GIVEN_HIGH_TELL = 0.75 * 7.0 + 0.25 * 2.0
MEAN_GIVEN_LOW_TELL = 0.25 * 7.0 + 0.75 * 2.0


def own_view(obs: dict) -> tuple[list[int], list[bool]]:
    """Values and known-flags of the observer's own hand."""
    values = [int(v) for v in obs["board_views"][obs["observer"]]["values"]]
    return values, [v != UNKNOWN_VALUE for v in values]


def tell_estimates(obs: dict) -> dict[int, list[float]]:
    """
    Per-player, per-position card estimates from public information only.

    Replays the public history: a tell (PEEK or KEEP) sets the estimate for the
    card that was looked at, and a TRADE moves estimates between hands along
    with the cards. Cards the observer knows use their true value.
    """
    me = obs["observer"]
    other = 1 - me
    est = {p: [PRIOR_MEAN] * HAND_SIZE for p in (me, other)}
    for event in obs["history"]:
        actor, pos = event["actor"], event["position"]
        if event["type"] in ("PEEK", "KEEP"):
            est[actor][pos] = MEAN_GIVEN_HIGH_TELL if event["tell"] == "HIGH" else MEAN_GIVEN_LOW_TELL
        elif event["type"] == "TRADE":
            given = event["own_position"]
            est[actor][given], est[1 - actor][pos] = est[1 - actor][pos], est[actor][given]
    for p in (me, other):
        for pos, value in enumerate(obs["board_views"][p]["values"]):
            if value != UNKNOWN_VALUE:
                est[p][pos] = float(value)
    return est


@dataclass
class Bot:
    seed: int = 0
    epsilon: float = 0.0
    rng: random.Random = field(init=False, repr=False)

    name = "bot"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_action(self, obs: dict, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)
        action = self._choose(obs, legal_actions)
        return action if action in legal_actions else self.rng.choice(legal_actions)

    def _choose(self, obs: dict, legal: list[int]) -> int:
        raise NotImplementedError


class RandomBot(Bot):
    name = "random"

    def _choose(self, obs: dict, legal: list[int]) -> int:
        return self.rng.choice(legal)


class GreedyBot(Bot):
    """Improve your own hand with the draws; call when it is fully known and low."""

    name = "greedy"
    keep_margin = 1.0  # keep a draw if it beats your worst card by at least this much
    call_threshold = 8

    def _choose(self, obs: dict, legal: list[int]) -> int:
        values, known = own_view(obs)
        drawn = obs["drawn_card"]
        worst = obs["own_worst_slot"]
        worst_value = values[worst] if known[worst] else PRIOR_MEAN

        if CALL_ACTION in legal and all(known) and sum(values) <= self.call_threshold:
            return CALL_ACTION
        if KEEP_ACTION in legal and drawn <= worst_value - self.keep_margin:
            return KEEP_ACTION
        if PEEK_ACTION in legal:
            return PEEK_ACTION
        if CALL_ACTION in legal:
            return CALL_ACTION
        if KEEP_ACTION in legal and drawn < worst_value:
            return KEEP_ACTION
        return self.rng.choice(legal)


class TellReaderBot(Bot):
    """Scores every action by its expected swing in (opponent total - own total) under tell-based estimates."""

    name = "tell_reader"
    peek_value = 1.5
    call_margin = 3.0

    def _choose(self, obs: dict, legal: list[int]) -> int:
        me = obs["observer"]
        est = tell_estimates(obs)
        mine, theirs = est[me], est[1 - me]
        worst_value = mine[obs["own_worst_slot"]]
        drawn = obs["drawn_card"]

        if CALL_ACTION in legal and sum(mine) + self.call_margin <= sum(theirs):
            return CALL_ACTION
        scored: list[tuple[float, int]] = []
        if KEEP_ACTION in legal:
            scored.append((worst_value - drawn, KEEP_ACTION))
        if PEEK_ACTION in legal:
            scored.append((self.peek_value, PEEK_ACTION))
        for j in range(HAND_SIZE):
            if TRADE_BASE + j in legal:
                # Trading moves the difference to both totals.
                scored.append((2.0 * (worst_value - theirs[j]), TRADE_BASE + j))
        best_score, best_action = max(scored) if scored else (0.0, self.rng.choice(legal))
        if best_score <= 0.0 and CALL_ACTION in legal and sum(mine) < sum(theirs):
            return CALL_ACTION
        return best_action


BOT_REGISTRY: dict[str, type[Bot]] = {
    RandomBot.name: RandomBot,
    GreedyBot.name: GreedyBot,
    TellReaderBot.name: TellReaderBot,
}


def make_bot(name: str, seed: int = 0, epsilon: float = 0.0) -> Bot:
    if name not in BOT_REGISTRY:
        raise KeyError(f"Unknown bot {name!r}. Choose from {sorted(BOT_REGISTRY)}.")
    return BOT_REGISTRY[name](seed=seed, epsilon=epsilon)
