"""MuZero training loop for Hidden Hand.

    python -m muzero.train --out runs/muzero --iterations 1000

Each iteration plays ``--games-per-iter`` games with the current network, adds them to the
replay buffer and takes ``--grad-steps`` gradient steps on unrolled mini-batches. The per-iteration
losses go to ``<out>/history.json`` (the format ``python -m hidden_hand.checks`` reads), the network
to ``<out>/model.pt`` and, every ``--ckpt-every`` iterations, to ``<out>/ckpt_<iteration>.pt``.

Curriculum: each game is self-play with probability ``TrainConfig.self_play_prob(iteration)``,
otherwise the network plays one seat against a bot drawn from ``bot_pool`` (``bot_weights``). The
self-play probability holds at ``self_play_start`` until ``self_play_ramp[0]`` of training, rises
linearly to ``self_play_end`` at ``self_play_ramp[1]`` and stays there.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hidden_hand import make_bot

from .mcts import MCTSConfig
from .model import ModelConfig, MuZeroNet, scalar_to_support, scale_gradient
from .selfplay import ReplayBuffer, play_games


@dataclass
class TrainConfig:
    iterations: int = 1000
    games_per_iter: int = 256
    grad_steps: int = 96
    batch_size: int = 64
    unroll: int = 5
    td_steps: int = 20            # longer than a game: pure Monte-Carlo return targets
    discount: float = 1.0
    reward_scale: float = 10.0    # rewards (minus the final score) are divided by this
    buffer_games: int = 5000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    policy_weight: float = 1.0
    value_weight: float = 1.0
    reward_weight: float = 1.0
    sims_schedule: tuple[int, ...] = (8, 12, 16)  # equal thirds of training
    # bot / self-play curriculum
    self_play_start: float = 0.25                 # P(self-play game) early in training
    self_play_end: float = 0.9                    # P(self-play game) late in training
    self_play_ramp: tuple[float, float] = (0.3, 0.7)  # fractions of training where the ramp starts / ends
    bot_pool: tuple[str, ...] = ("random", "greedy", "tell_reader")
    bot_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    ckpt_every: int = 50
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)

    def simulations(self, iteration: int) -> int:
        stage = min(len(self.sims_schedule) - 1, iteration * len(self.sims_schedule) // max(1, self.iterations))
        return self.sims_schedule[stage]

    def self_play_prob(self, iteration: int) -> float:
        frac = iteration / max(1, self.iterations)
        lo, hi = self.self_play_ramp
        if frac <= lo:
            return self.self_play_start
        if frac >= hi:
            return self.self_play_end
        return self.self_play_start + (self.self_play_end - self.self_play_start) * (frac - lo) / (hi - lo)


def sample_opponents(cfg: TrainConfig, iteration: int, num_games: int, rng: np.random.Generator, seed0: int) -> list:
    """``None`` (self-play) or a fresh bot for each game of this iteration."""
    p_self = cfg.self_play_prob(iteration)
    weights = np.asarray(cfg.bot_weights, dtype=np.float64)
    weights = weights / weights.sum()
    opponents = []
    for i in range(num_games):
        if rng.random() < p_self:
            opponents.append(None)
        else:
            opponents.append(make_bot(cfg.bot_pool[rng.choice(len(cfg.bot_pool), p=weights)], seed=seed0 + i))
    return opponents


def save_checkpoint(net: MuZeroNet, path: Path, iteration: int) -> None:
    torch.save({"model_config": net.cfg.to_dict(), "state_dict": net.state_dict(), "iteration": iteration}, path)


def load_checkpoint(path: str | Path) -> MuZeroNet:
    data = torch.load(path, map_location="cpu", weights_only=True)
    net = MuZeroNet(ModelConfig(**data["model_config"]))
    net.load_state_dict(data["state_dict"])
    return net.eval()


def train_step(net: MuZeroNet, opt: torch.optim.Optimizer, batch: dict, cfg: TrainConfig) -> dict[str, float]:
    t = {k: torch.from_numpy(v) for k, v in batch.items()}
    K = cfg.unroll
    S = net.cfg.support_size
    state, logits, value = net.initial_inference(t["feats"], t["pad"], raw=True)

    def categorical_ce(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of ``[B, 2, bins]`` logits against two-hot targets of ``[B, 2]`` scalars,
        summed over the two entries -> ``[B]``."""
        ce = F.cross_entropy(pred_logits.transpose(1, 2), scalar_to_support(target, S).transpose(1, 2), reduction="none")
        return ce.sum(-1)

    policy_loss = value_loss = reward_loss = 0.0
    for k in range(K + 1):
        if k > 0:
            state, reward, logits, value = net.recurrent_inference(state, t["actions"][:, k - 1], raw=True)
            state = scale_gradient(state, 0.5)
            reward_loss = reward_loss + categorical_ce(reward, t["reward"][:, k]).mean() / K
        ce = F.cross_entropy(logits, t["policy"][:, k], reduction="none")
        weight = 1.0 if k == 0 else 1.0 / K
        policy_loss = policy_loss + weight * (ce * t["policy_mask"][:, k]).mean()
        value_loss = value_loss + weight * categorical_ce(value, t["value"][:, k]).mean()

    loss = cfg.policy_weight * policy_loss + cfg.value_weight * value_loss + cfg.reward_weight * reward_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
    opt.step()
    return {"loss_policy": policy_loss.item(), "loss_value": value_loss.item(), "loss_reward": reward_loss.item()}


def train(cfg: TrainConfig, out: Path, log=print) -> MuZeroNet:
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = MuZeroNet(cfg.model)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.buffer_games, cfg.td_steps, cfg.discount)
    history: list[dict] = []
    seed_counter = cfg.seed * 10_000_000

    for it in range(cfg.iterations):
        t0 = time.time()
        sims = cfg.simulations(it)
        opponents = sample_opponents(cfg, it, cfg.games_per_iter, rng, seed_counter)
        net.eval()
        games = play_games(
            net,
            cfg.games_per_iter,
            MCTSConfig(num_simulations=sims, discount=cfg.discount),
            rng,
            seed0=seed_counter,
            reward_scale=cfg.reward_scale,
            temperature=lambda step: 1.0 if step < 8 else 0.5,
            opponents=opponents,
        )
        seed_counter += cfg.games_per_iter
        buffer.add(games)
        t_self = time.time() - t0

        net.train()
        losses = [train_step(net, opt, buffer.sample_batch(cfg.batch_size, cfg.unroll, rng), cfg)
                  for _ in range(cfg.grad_steps)]
        row = {k: float(np.mean([l[k] for l in losses])) for k in losses[0]}
        self_games = [g for g in games if g.opponent == "self"]
        bot_games = [g for g in games if g.opponent != "self"]
        row.update(
            iteration=it + 1,
            simulations=sims,
            self_play_prob=cfg.self_play_prob(it),
            self_play_games=len(self_games),
            mean_score=float(np.mean([g.final_scores[0] + g.final_scores[1] for g in games]) / 2),
            # network's score vs the bots this iteration (win 1, draw 0.5); None if no bot games
            bot_score=(float(np.mean([1.0 if g.winner == g.net_seat else 0.5 if g.winner == -1 else 0.0
                                      for g in bot_games])) if bot_games else None),
            mean_game_length=float(np.mean([len(g) for g in games])),
            selfplay_seconds=t_self,
            seconds=time.time() - t0,
        )
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        save_checkpoint(net, out / "model.pt", it + 1)
        if (it + 1) % cfg.ckpt_every == 0:
            save_checkpoint(net, out / f"ckpt_{it + 1}.pt", it + 1)
        bot_txt = f" vs-bots={row['bot_score']:.2f}" if bot_games else ""
        log(f"it {it + 1:4d}/{cfg.iterations} sims={sims:2d} self-play={len(self_games)}/{len(games)}{bot_txt} policy={row['loss_policy']:.3f} "
            f"value={row['loss_value']:.3f} reward={row['loss_reward']:.4f} "
            f"score={row['mean_score']:.1f} len={row['mean_game_length']:.1f} "
            f"({t_self:.0f}s self-play, {row['seconds']:.0f}s total)")
    return net.eval()


def main() -> None:
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/muzero")
    p.add_argument("--iterations", type=int, default=d.iterations)
    p.add_argument("--games-per-iter", type=int, default=d.games_per_iter)
    p.add_argument("--grad-steps", type=int, default=d.grad_steps)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--unroll", type=int, default=d.unroll)
    p.add_argument("--buffer-games", type=int, default=d.buffer_games)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--ckpt-every", type=int, default=d.ckpt_every)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--self-play-start", type=float, default=d.self_play_start,
                   help="probability a game is self-play early in training (the rest are vs bots)")
    p.add_argument("--self-play-end", type=float, default=d.self_play_end,
                   help="probability a game is self-play late in training")
    p.add_argument("--self-play-ramp", type=float, nargs=2, default=d.self_play_ramp, metavar=("START", "END"),
                   help="fractions of training over which the self-play probability ramps up")
    p.add_argument("--bots", nargs="+", default=list(d.bot_pool), help="bot pool for non-self-play games")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = TrainConfig(
        iterations=args.iterations, games_per_iter=args.games_per_iter, grad_steps=args.grad_steps,
        batch_size=args.batch_size, unroll=args.unroll, buffer_games=args.buffer_games, lr=args.lr,
        ckpt_every=args.ckpt_every, seed=args.seed,
        self_play_start=args.self_play_start, self_play_end=args.self_play_end,
        self_play_ramp=tuple(args.self_play_ramp), bot_pool=tuple(args.bots), bot_weights=(1.0,) * len(args.bots),
    )
    train(cfg, Path(args.out))


if __name__ == "__main__":
    main()
