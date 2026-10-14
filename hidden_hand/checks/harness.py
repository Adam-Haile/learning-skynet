"""
Quality-check harness: "what a correct implementation looks like" on Hidden Hand.

You bring two agents (your MuZero and your SkyNet) and, optionally, their
training logs. The harness plays games, runs the checks tier by tier and
reports each result next to the target range from the mentor's reference runs.

Agent interface (duck-typed; see README.md):

    class MyAgent:
        name = "my-skynet"
        def act(self, obs: dict, legal_actions: list[int]) -> int: ...
        # optional, much faster: one call for many games at once
        def act_batch(self, observations: list[dict], legal_actions: list[list[int]]) -> list[int]: ...
        # optional, needed for the Tier 4 probes: frozen latent per observation, shape [N, D].
        # For SkyNet return the ego-conditioned latent with ego = current = obs["observer"].
        def latent(self, observations: list[dict]) -> numpy.ndarray: ...
        # optional, SkyNet only, needed for the Tier 4 ablation: the same weights with the
        # conditioning changed at inference; mode is "zero_ego", "ego_only" or "none".
        def with_conditioning(self, mode: str) -> "MyAgent": ...

Agents should play at evaluation strength: greedy MCTS (temperature ~0),
no Dirichlet noise.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from hidden_hand.bots import Bot, make_bot
from hidden_hand.checks.stats import ridge_cv_r2, wilson_interval
from hidden_hand.env import HiddenHandConfig, HiddenHandEnv

DEFAULT_TARGETS = Path(__file__).with_name("targets.json")
MAX_DECISIONS_PER_GAME = 200  # far above the rules' limit; anything longer counts as a truncation


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
@dataclass
class CheckResult:
    tier: int
    name: str
    passed: bool | None  # None = skipped
    value: str
    target: str
    detail: str = ""


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def all_passed(self) -> bool:
        return all(r.passed is not False for r in self.results)

    def table(self) -> str:
        rows = [("tier", "check", "result", "value", "target")]
        for r in self.results:
            status = "SKIP" if r.passed is None else ("PASS" if r.passed else "FAIL")
            rows.append((str(r.tier), r.name, status, r.value, r.target))
        widths = [max(len(row[i]) for row in rows) for i in range(5)]
        lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows]
        lines.insert(1, "  ".join("-" * w for w in widths))
        details = [f"  - {r.name}: {r.detail}" for r in self.results if r.detail]
        return "\n".join(lines + ([""] + details if details else []))

    def to_json(self) -> list[dict]:
        return [r.__dict__ for r in self.results]


# ----------------------------------------------------------------------
# Playing games
# ----------------------------------------------------------------------
class _BotAgent:
    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self.name = bot.name

    def act(self, obs: dict, legal_actions: list[int]) -> int:
        return self.bot.select_action(obs, legal_actions)


@dataclass
class MatchResult:
    games: int
    wins: int
    draws: int
    losses: int
    illegal_actions: int
    truncated: int
    mean_length: float
    observations: list[dict] = field(default_factory=list)  # ordered game by game when recorded
    hidden: dict[str, list[float]] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(1, self.games)

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.wins + 0.5 * self.draws, self.games)


def _act(agent: Any, observations: list[dict], legal: list[list[int]]) -> list[int]:
    if hasattr(agent, "act_batch"):
        return [int(a) for a in agent.act_batch(observations, legal)]
    return [int(agent.act(o, l)) for o, l in zip(observations, legal)]


def play_match(
    agent_a: Any,
    agent_b: Any,
    games: int,
    seed: int = 0,
    env_config: HiddenHandConfig | None = None,
    record_observations: bool = False,
) -> MatchResult:
    """
    A vs B with alternating seats (A is seat 0 in even games). Games run in
    lockstep so agents with ``act_batch`` get one call per player per round.
    Illegal actions are counted and replaced by a random legal action.
    """
    fix_rng = random.Random(seed)
    envs = [HiddenHandEnv(seed=seed + g, config=env_config) for g in range(games)]
    seats = [(agent_a, agent_b) if g % 2 == 0 else (agent_b, agent_a) for g in range(games)]
    lengths = [0] * games
    illegal = truncated = 0
    recorded: list[tuple[int, dict, dict]] = []  # (game, observation, hidden features)
    active = list(range(games))
    while active:
        by_agent: dict[int, list[int]] = {}
        for g in active:
            by_agent.setdefault(id(seats[g][envs[g].current_player]), []).append(g)
        chosen: dict[int, int] = {}
        for ids in by_agent.values():
            agent = seats[ids[0]][envs[ids[0]].current_player]
            obs = [envs[g].observe(envs[g].current_player) for g in ids]
            legal = [envs[g].legal_actions() for g in ids]
            if record_observations:
                recorded.extend((g, o, envs[g].hidden_features(envs[g].current_player)) for g, o in zip(ids, obs))
            for g, action, lg in zip(ids, _act(agent, obs, legal), legal):
                if action not in lg:
                    illegal += 1
                    action = fix_rng.choice(lg)
                chosen[g] = action
        next_active = []
        for g in active:
            _, _, done, _ = envs[g].step(chosen[g])
            lengths[g] += 1
            if done:
                continue
            if lengths[g] >= MAX_DECISIONS_PER_GAME:
                truncated += 1
                continue
            next_active.append(g)
        active = next_active

    # Order rows game by game (stable), so the probe's contiguous CV folds hold out whole
    # games instead of slices of game time.
    recorded.sort(key=lambda row: row[0])
    observations = [o for _, o, _ in recorded]
    hidden: dict[str, list[float]] = {}
    for _, _, feats in recorded:
        for k, v in feats.items():
            hidden.setdefault(k, []).append(v)

    wins = draws = 0
    for g, env in enumerate(envs):
        if env.game_over and env.winner == -1:
            draws += 1
        elif env.game_over and env.winner == g % 2:
            wins += 1
    return MatchResult(
        games=games,
        wins=wins,
        draws=draws,
        losses=games - wins - draws,
        illegal_actions=illegal,
        truncated=truncated,
        mean_length=float(np.mean(lengths)),
        observations=observations,
        hidden=hidden,
    )


# ----------------------------------------------------------------------
# Targets and logs
# ----------------------------------------------------------------------
def load_targets(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_TARGETS).read_text())


def load_training_log(path: str | Path) -> list[dict[str, float]]:
    """A JSON list of per-iteration dicts, or a CSV with one row per iteration."""
    path = Path(path)
    if path.suffix == ".csv":
        with path.open() as f:
            return [{k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(f)]
    data = json.loads(path.read_text())
    if isinstance(data, dict):  # {"loss_policy": [...], ...}
        n = max(len(v) for v in data.values())
        return [{k: v[i] for k, v in data.items() if i < len(v)} for i in range(n)]
    return data


def _fmt_range(t: dict) -> str:
    return f"[{t['min']:.2f}, {t['max']:.2f}]"


def _window_mean(values: list[float], start_frac: float, end_frac: float) -> float:
    n = len(values)
    lo, hi = int(n * start_frac), max(int(n * start_frac) + 1, int(n * end_frac))
    window = [v for v in values[lo:hi] if v is not None and math.isfinite(v)]
    return float(np.mean(window)) if window else math.nan


# ----------------------------------------------------------------------
# Tiers
# ----------------------------------------------------------------------
def check_losses(report: Report, label: str, log: list[dict] | None, keys: Iterable[str], targets: dict) -> None:
    """Tier 1: each loss's mean over the last 20% of training vs its first 20% (after a short warm-up)."""
    for key in keys:
        name = f"{label} {key} decreases"
        target = targets.get(key, {"max_ratio": 1.0})
        if log is None:
            report.add(CheckResult(1, name, None, "-", f"last/first <= {target['max_ratio']:.2f}", "no training log given"))
            continue
        series = [row.get(key) for row in log if key in row]
        if len(series) < 10:
            report.add(CheckResult(1, name, False, f"{len(series)} rows", "need >= 10 iterations", f"missing {key!r}?"))
            continue
        first, last = _window_mean(series, 0.02, 0.22), _window_mean(series, 0.8, 1.0)
        ratio = last / first if first else math.inf
        report.add(
            CheckResult(1, name, ratio <= target["max_ratio"], f"{first:.3f} -> {last:.3f} ({ratio:.2f}x)",
                        f"last/first <= {target['max_ratio']:.2f}")
        )


def check_plays_legally(report: Report, label: str, agent: Any, games: int, seed: int, env_config) -> None:
    """Tier 1: only legal actions across N games; every game ends on its own."""
    result = play_match(agent, _BotAgent(make_bot("random", seed=seed)), games, seed=seed, env_config=env_config)
    report.add(CheckResult(1, f"{label} legal actions only", result.illegal_actions == 0,
                           f"{result.illegal_actions} illegal in {games} games", "0"))
    report.add(CheckResult(1, f"{label} episodes terminate", result.truncated == 0,
                           f"{result.truncated} truncated, mean length {result.mean_length:.1f}", "0 truncated"))


def _pass_min(target: dict) -> float:
    return target.get("pass_min", target["min"])


def check_vs_bots(report: Report, tier: int, label: str, agent: Any, targets: dict, games: int, seed: int, env_config) -> None:
    """Tiers 2/3: score against the bundled bots, with the reference range shown next to the threshold."""
    for i, (bot_name, target) in enumerate(targets.items()):
        bot = _BotAgent(make_bot(bot_name, seed=seed + i))
        result = play_match(agent, bot, games, seed=seed + 1000 * (i + 1), env_config=env_config)
        lo, hi = result.ci
        ok = result.score >= _pass_min(target)
        report.add(
            CheckResult(tier, f"{label} vs {bot_name}", ok,
                        f"{result.score:.3f} (95% CI {lo:.3f}-{hi:.3f})",
                        f">= {_pass_min(target):.2f}; reference {_fmt_range(target)}",
                        "" if ok else "below what a correct build reaches; check training length and evaluation settings")
        )


def check_head_to_head(report: Report, skynet: Any, muzero: Any, target: dict, games: int, seed: int, env_config) -> None:
    """Tier 3: SkyNet beats MuZero at a matched budget; the Wilson 95% CI must sit above 50%."""
    result = play_match(skynet, muzero, games, seed=seed, env_config=env_config)
    lo, hi = result.ci
    report.add(
        CheckResult(3, "SkyNet vs MuZero head-to-head", lo > 0.5,
                    f"{result.score:.3f} (95% CI {lo:.3f}-{hi:.3f}, {games} games)",
                    f"CI above 0.50; reference {_fmt_range(target)}, mean {target['ref_mean']:.2f}",
                    f"W/D/L = {result.wins}/{result.draws}/{result.losses}")
    )


def check_probes(report: Report, skynet: Any, muzero: Any, targets: dict, games: int, seed: int, env_config) -> None:
    """Tier 4: ridge probes (5-fold CV) on frozen latents from the same decision steps."""
    if not (hasattr(skynet, "latent") and hasattr(muzero, "latent")):
        report.add(CheckResult(4, "linear probes", None, "-", "-", "agents need a latent(observations) method"))
        return
    match = play_match(skynet, muzero, games, seed=seed, env_config=env_config, record_observations=True)
    obs = match.observations
    reps = {"SkyNet": np.asarray(skynet.latent(obs)), "MuZero": np.asarray(muzero.latent(obs))}
    for feature, spec in targets.items():
        y = np.asarray(match.hidden[feature])
        for rep_name, x in reps.items():
            t = spec[rep_name.lower()]
            r2 = ridge_cv_r2(x, y)
            report.add(CheckResult(4, f"probe {feature} [{spec['kind']}] {rep_name}", r2 >= t["min_r2"], f"R2={r2:.3f}",
                                   f">= {t['min_r2']:.2f} (ref {t['ref_min']:.2f}..{t['ref_max']:.2f})"))
    report.results[-1].detail = f"{len(obs)} decision steps from {games} SkyNet-vs-MuZero games"


ABLATION_MODES = ("zero_ego", "ego_only", "none")


def check_ablation(report: Report, skynet: Any, targets: dict, games: int, seed: int, env_config) -> None:
    """
    Tier 4: inference-time conditioning ablation (paper Table 2).

    The full SkyNet plays the *same weights* with part of the conditioning removed:
    "zero_ego" (keep current-player and player-count embeddings, drop the ego embedding),
    "ego_only" (keep only the ego embedding) and "none" (no conditioning). The paper's
    ordering, which the reference reproduces, is full > zero_ego >> ego_only ~ none.
    """
    if not hasattr(skynet, "with_conditioning"):
        report.add(CheckResult(4, "conditioning ablation", None, "-", "full > zero_ego >> ego_only ~ none",
                               "SkyNet agent needs with_conditioning(mode) returning an ablated agent"))
        return
    scores = {}
    for i, mode in enumerate(ABLATION_MODES):
        result = play_match(skynet, skynet.with_conditioning(mode), games, seed=seed + i, env_config=env_config)
        scores[mode] = result.score
        if mode != "zero_ego":
            report.add(CheckResult(4, f"ablation: full vs {mode}", result.score >= _pass_min(targets[mode]),
                                   f"{result.score:.3f}",
                                   f">= {_pass_min(targets[mode]):.2f}; reference {_fmt_range(targets[mode])}"))
    hurts_least = scores["zero_ego"] < (scores["ego_only"] + scores["none"]) / 2
    report.add(CheckResult(4, "ablation ordering", hurts_least,
                           " / ".join(f"{m} {scores[m]:.2f}" for m in ABLATION_MODES),
                           "zero_ego hurts less than the average of ego_only and none",
                           f"reference zero_ego {_fmt_range(targets['zero_ego'])}"))


def run_checks(
    muzero: Any,
    skynet: Any,
    muzero_log: str | Path | list[dict] | None = None,
    skynet_log: str | Path | list[dict] | None = None,
    targets: dict | str | Path | None = None,
    h2h_games: int = 600,
    bot_games: int = 300,
    legality_games: int = 100,
    probe_games: int = 300,
    ablation_games: int = 300,
    seed: int = 12345,
    env_config: HiddenHandConfig | None = None,
    tiers: Iterable[int] = (1, 2, 3, 4),
    progress: Callable[[str], None] | None = print,
) -> Report:
    """Run every tier and return a Report (``print(report.table())``)."""
    t = targets if isinstance(targets, dict) else load_targets(targets)
    tiers = set(tiers)
    say = progress or (lambda _msg: None)
    logs = {
        "MuZero": load_training_log(muzero_log) if isinstance(muzero_log, (str, Path)) else muzero_log,
        "SkyNet": load_training_log(skynet_log) if isinstance(skynet_log, (str, Path)) else skynet_log,
    }
    report = Report()
    if 1 in tiers:
        say("Tier 1: it runs")
        check_losses(report, "MuZero", logs["MuZero"], ("loss_policy", "loss_value", "loss_reward"), t["losses"])
        check_losses(report, "SkyNet", logs["SkyNet"],
                     ("loss_policy", "loss_value", "loss_reward", "loss_winner", "loss_rank"), t["losses"])
        check_plays_legally(report, "MuZero", muzero, legality_games, seed, env_config)
        check_plays_legally(report, "SkyNet", skynet, legality_games, seed + 1, env_config)
    if 2 in tiers:
        say("Tier 2: the baseline works")
        check_vs_bots(report, 2, "MuZero", muzero, t["muzero_vs_bots"], bot_games, seed + 2, env_config)
    if 3 in tiers:
        say("Tier 3: SkyNet works")
        check_vs_bots(report, 3, "SkyNet", skynet, t["skynet_vs_bots"], bot_games, seed + 3, env_config)
        check_head_to_head(report, skynet, muzero, t["head_to_head"], h2h_games, seed + 4, env_config)
    if 4 in tiers:
        say("Tier 4: the mechanism works")
        check_probes(report, skynet, muzero, t["probes"], probe_games, seed + 5, env_config)
        if "ablation" in t:
            check_ablation(report, skynet, t["ablation"], ablation_games, seed + 6, env_config)
    return report
