# Hidden Hand: a toy environment for rebuilding SkyNet

Hidden Hand is a tiny demo card game to showcase SkyNet's advantages and disadvantages. It keeps the
structure of Skyjo that SkyNet was designed for, at a size you can train in minutes:

- **two players**, alternating turns;
- **hidden cards** that decide who wins;
- **chance** (random draws);
- **noisy public signals** ("tells") that make the opponent's hand *partly* inferable
  from the action history.

Your job is to implement **MuZero** and **SkyNet (Belief-Aware MuZero)** yourselves,
from the paper, train both here, and run the quality checks. The checks describe what a
correct implementation looks like on this game, measured against the mentor's reference
runs. They are not a grade. When your build disagrees with the paper, the repo or the
checks, write it down: those notes feed the reproducibility section of our paper.

- Paper: Haile, *SkyNet: Belief-Aware Planning in Partially Observable Stochastic Games*,
  ICANN 2026. https://doi.org/10.1007/978-3-032-38404-1_9
- Repo: https://github.com/DevArtech/skynet (read it *after* your own first build, then diff)

## Install

```bash
pip install -e ".[dev]"
pytest
```

Only `numpy` is required by the environment and the checks. Bring whatever you use for
models (PyTorch is what the repo uses).

## Rules

Each player is dealt 4 face-down cards from a deck with four each of 0–9, and privately
knows 1 of the 4 at the start. **Lowest total wins.**

The **draw pile** is separate and holds only high cards (four each of 5–9). Drawing rarely
improves your hand. The low cards are in the two hands, and the way to get one is to take
it from your opponent.

Your **worst card** is the one you would most like to get rid of: your highest known card,
or one of your unknown cards if every known card is below 4.5. Unknown cards count as 4.5.

Every turn begins with a **chance event**: you draw the top card of the draw pile and look
at it privately. Then you take exactly one action:

| id | action | effect |
|---|---|---|
| 0 | `KEEP` | put the drawn card in place of your worst card; the old card goes face-up on the discard pile |
| 1 | `PEEK` | privately look at one of your unknown cards (legal only if you have one) |
| 2–5 | `TRADE(j)` | blindly swap your worst card with the opponent's card at position `j` |
| 6 | `CALL` | end the game now (legal from your 3rd turn) |

- Unless you `KEEP`, the drawn card is discarded face-up.
- **Tells.** Whenever someone looks at a card in their hand (`PEEK`, or the card they
  `KEEP`), the table sees a tell: `HIGH` if the card is ≥ 5, otherwise `LOW`. Each tell
  is wrong 25% of the time. Looking at your own cards therefore leaks where your low
  cards are.
- **Trades are blind.** Nobody looks at anything; whoever knew a card keeps knowing it. After
  a trade you know the card you gave away, and you do not know the card you received.
- The game ends on a `CALL`, or after both players have taken 8 turns (at most 16 decisions).
- **Caller penalty (as in Skyjo).** If the caller is not strictly lowest, their score doubles.
- Ties are draws.

Why the belief matters: picking *which* opponent card to trade for depends on what you
believe about the opponent's hidden cards. A bot that reads tells and trades beats one
that only manages its own hand about 3 games in 4.

## Using the environment

```python
from hidden_hand import HiddenHandEnv, make_bot

env = HiddenHandEnv(seed=0)          # deterministic under a seed
obs = env.reset()                     # observation for the player to move
while not env.game_over:
    legal = env.legal_actions()       # or env.legal_action_mask() (length 7)
    action = legal[0]
    obs, rewards, done, info = env.step(action)
print(env.scores, env.winner)         # winner = -1 for a draw
print(env.render())                   # full table (render(observer=p) for one player's view)
```

The API mirrors the repo's `SkyjoEnv`: `step()` returns
`(next_obs, rewards, terminated, info)`, where `next_obs` is for the next player to move
and `rewards` is `{"player_0": r0, "player_1": r1}`. Illegal actions raise `ValueError`.

### Observation

`env.observe(p)` is a dict. The model-facing part is `obs["tokens"]`, in the repo's
structured schema, so an encoder built for `skyjo_env.py` works with small config changes:

| token | shape | fields |
|---|---|---|
| `board_tokens` | 8 × 4 | `[owner, position, visible, value]`, observer's board first; `visible` means *known to the observer*; unknown values are `UNKNOWN_VALUE = -99` |
| `discard_token` | 2 | `[discard_top, discard_size]` |
| `global_token` | 6 | `[deck_size, step, current_player, phase, phase_bucket, round_index]` |
| `decision_token` | 2 | `[sees_draw, drawn_value]`: the drawn card, visible only to the player to move |
| `action_tokens` | ≤ 20 × 6 | public history `[actor, type, source, target_pos, a_value, b_value]` |

Action history types: 1 `KEEP`, 2 `PEEK`, 3 `TRADE`, 4 `CALL`. `a_value` is the tell
(0 `LOW`, 1 `HIGH`, −127 none). `b_value` is the card that went face-up on the discard
pile (−127 none). For `TRADE`, `source` is the trader's own position and `target_pos` is
the opponent's position. The whole game fits in the 20-token history window.

The rest of the dict holds human-friendly versions of the same information
(`board_views`, `history`, `drawn_card`, `own_worst_slot`, ...).

### Rewards

The default `reward_mode="own_score"` is Skyjo's scheme: when the game ends, each player
receives **minus their own final score**. Nothing is paid before that. The game is
therefore **not zero-sum**, even though the winner is still whoever scores lower: a trade
that helps you by 5 does not show up as −5 in the opponent's reward for that move, it
shows up in their own final score.

This matters for your implementation. As in the paper, a value is a player's expected
return, and search is used unchanged: do not negate values between players, because one
player's return is not the negative of the other's.

Other modes exist for experiments (`"shaped"`, `"score_diff"`, `"win_loss"`, all zero-sum),
and `HiddenHandConfig(draw_pile_values=None)` draws from the rest of the dealt deck
instead of the high-card pile. The reference numbers assume the defaults.

### Ground truth for probes, never for training

`info["hidden"]` and `env.hidden_features(p)` expose true quantities such as the
opponent's hidden total, player `p`'s unknown cards and the true score difference. They exist for
the Tier 4 linear probes. **Never feed them to your model.** Nothing hidden is in the
observation.

### Bots

`make_bot(name, seed)` returns `random`, `greedy` (manages its own hand, never trades,
ignores the opponent) or `tell_reader` (tracks tells through trades, trades for the
opponent card that looks lowest, calls when it believes it is ahead). Each has
`select_action(obs, legal_actions)`.

## Running the quality checks

Wrap each trained model in a small agent:

```python
class MyAgent:
    name = "my-skynet"

    def act(self, obs, legal_actions) -> int: ...
    # optional but much faster: one call for many games in lockstep
    def act_batch(self, observations, legal_actions) -> list[int]: ...
    # Tier 4 probes: frozen latent per observation, array [N, D].
    # MuZero: the representation output h. SkyNet: the ego-conditioned latent,
    # with ego = current player = obs["observer"].
    def latent(self, observations) -> "np.ndarray": ...
    # Tier 4 ablation (SkyNet only): the same weights with the conditioning changed
    # at inference. mode is "zero_ego", "ego_only" or "none" (paper, Section 6.3).
    def with_conditioning(self, mode) -> "MyAgent": ...
```

Evaluate at full strength: greedy MCTS (temperature ≈ 0), no Dirichlet noise, 16
simulations. Then run:

```bash
python -m hidden_hand.checks \
    --muzero my_pkg.agents:load_muzero --skynet my_pkg.agents:load_skynet \
    --muzero-log runs/muzero/history.json --skynet-log runs/skynet/history.json
```

or, from Python, `print(run_checks(muzero_agent, skynet_agent, muzero_log, skynet_log).table())`.
The full run takes a few minutes with `act_batch`.

Training logs are a JSON list with one dict per iteration, or a CSV, with keys
`loss_policy`, `loss_value`, `loss_reward`, plus `loss_winner` and `loss_rank` for
SkyNet. Compare the two models at a **matched training budget**.

### What the checks are

| tier | check | pass condition |
|---|---|---|
| 1 | losses decrease | mean of the last 20% of iterations ≤ the first 20% (the winner and rank losses are allowed to stay about level) |
| 1 | legality / termination | 0 illegal actions and 0 truncated games over 100 games each |
| 2 | MuZero vs `random`, `greedy`, `tell_reader` | score at or above the threshold below |
| 3 | SkyNet vs the same bots | score at or above the threshold below |
| 3 | **SkyNet vs MuZero**, 600 games, alternating seats | SkyNet ahead: Wilson 95% interval entirely above 0.50 |
| 4 | conditioning ablation, 300 games per mode | full SkyNet beats its `ego_only` and `none` ablations, and `zero_ego` hurts less than those two |
| 4 | linear probes (ridge, 5-fold CV, 300 games) | both latents decode the observable control and the opponent's hidden total |

"Score" counts a win as 1 and a draw as ½. Bot thresholds sit well below the reference so
an independent implementation with different hyperparameters has room.

### Reference numbers

These come from the mentor's reference runs: 5 seeds per model, 1000 iterations, each
iteration 256 self-play games and 96 gradient steps of batch 64, MCTS with 8 → 12 → 16
simulations during training and 16 at evaluation. All ten runs take about 5 hours sharing
one T4, with self-play on CPU cores. The numbers are also in
[`hidden_hand/checks/targets.json`](hidden_hand/checks/targets.json).

**SkyNet's score against MuZero at matched checkpoints** (mean of 5 seeds):

| iteration | 50 | 100 | 200 | 300 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|---|---|---|
| SkyNet score | 0.53 | 0.59 | 0.75 | 0.70 | 0.66 | 0.58 | 0.62 | 0.67 |

At iteration 1000 SkyNet scores **0.67** pooled over 5 seeds (per seed: 0.68, 0.69, 0.76, 0.66, 0.55).
The lead opens early, peaks around iteration 200, narrows in mid-training and recovers.

| matchup at iteration 1000 | reference range | passes at |
|---|---|---|
| MuZero vs `random` | 0.43–0.53 | ≥ 0.36 |
| MuZero vs `greedy` | 0.54–0.60 | ≥ 0.45 |
| MuZero vs `tell_reader` | 0.26–0.39 | ≥ 0.20 |
| SkyNet vs `random` | 0.43–0.67 | ≥ 0.37 |
| SkyNet vs `greedy` | 0.63–0.73 | ≥ 0.56 |
| SkyNet vs `tell_reader` | 0.27–0.46 | ≥ 0.22 |
| full SkyNet vs `zero_ego` ablation | 0.48–0.70 | below the average of the next two |
| full SkyNet vs `ego_only` ablation | 0.54–0.93 | ≥ 0.50 |
| full SkyNet vs `none` ablation | 0.59–0.90 | ≥ 0.51 |

| probe feature | kind | MuZero R² | SkyNet R² |
|---|---|---|---|
| `opp_hidden_sum` | hidden | 0.20–0.24 | 0.15–0.24 |
| `total_diff` | hidden | 0.34–0.43 | 0.26–0.44 |
| `own_known_sum` | observable control | 0.97–0.98 | 0.90–0.97 |

### What to expect, and what is still open

- **SkyNet should beat MuZero.** In the reference it wins about two games in three at
  the end of training and three in four at its peak. Four of the five reference seeds are
  clearly ahead at iteration 1000; one is only slightly ahead. Other budgets tried by the
  mentor (more simulations, or 300 iterations) still had SkyNet ahead at every checkpoint
  but by less, ending at 55–60%. Somewhere between 55% and 67% at the end of training is
  normal; the check only asks that SkyNet is clearly ahead. Check several checkpoints, not
  just one, and train the full 1000 iterations: a compressed 300-iteration schedule does
  not reproduce the long run's early peak.
- **SkyNet should also be the stronger player against the bots**, most visibly against
  `greedy`.
- **MuZero is weak on this game.** The reference MuZero is close to coin-flip level against
  `random`. That is why the Tier 2 thresholds are low: they confirm your MuZero matches the
  reference, not that it is strong. Why MuZero struggles here while SkyNet does not is the
  first thing to work out as a group, and whether the comparison is a fair one is a question
  to carry into the Scope phase.
- **The ablation ordering reproduces the paper** (full > zero-ego ≫ ego-only ≈ none) in
  four of five reference seeds.
- **The probes differ from the paper.** In the paper only the ego-conditioned latent decodes
  hidden cards. Here both latents decode the opponent's hidden total about equally. That is
  an open question too.
- **Throughput matters**, as the paper says. Too few self-play games per gradient step lets
  the auxiliary heads memorize game outcomes.

## Layout

```
hidden_hand/
  env.py        the game
  bots.py       random / greedy / tell_reader
  checks/       quality-check harness, stats (Wilson CI, ridge probe), targets.json
tests/          rule and interface tests
```
