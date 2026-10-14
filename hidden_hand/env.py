"""
Hidden Hand: a tiny stochastic, partially observable, two-player card game.

Hidden Hand keeps Skyjo's structure (face-down cards, random draws, a public
discard pile, lowest total wins, a penalty for ending the round when you are
not lowest) at a scale that trains in minutes on a CPU.

Rules (defaults in ``HiddenHandConfig``):

* Each player is dealt 4 face-down cards from four copies of each value 0..9 and
  privately knows 1 of them at the start. The draw pile is separate and holds only
  high cards (four copies each of 5..9), so drawing rarely improves a hand: the low
  cards are in the two hands, and the way to get them is to take them.
* Your "worst card" is the card in your hand you would most like to get rid
  of: your highest known card, or an unknown card if all your known cards
  are below the deck average (4.5). Unknown cards count as 4.5.
* Players alternate turns. A turn starts with a chance event: the player
  draws the top card of the draw pile and looks at it privately. Then they take
  exactly one action:
    - KEEP:     put the drawn card face down in place of your worst card. The
                old card goes face-up on the discard pile (its value is public).
    - PEEK:     privately look at one of your unknown cards.
    - TRADE(j): blindly exchange your worst card with the opponent's card at
                position j. Nobody looks at anything; whoever knew a card keeps
                knowing it, so after a trade you know the card you gave away.
    - CALL:     end the game immediately.
  Unless the action was KEEP, the drawn card is discarded face-up.
* Every time a player looks at a card in their hand (PEEK, or the card they
  KEEP), the table sees a noisy "tell": HIGH if the card is >= 5, LOW
  otherwise, but the tell is wrong with probability 0.25. Tells are the noisy
  public signal that makes the opponent's hand partly inferable from the
  action history. Looking at your own cards therefore leaks where your low
  cards are, and choosing which opponent card to TRADE for is where reading
  the opponent pays off.
* The game also ends when both players have taken 8 turns.
* Score = sum of your 4 cards. If the caller is not strictly lowest, the
  caller's score is doubled. Lowest score wins; equal scores are a draw.

Rewards (``reward_mode``): the default "own_score" is Skyjo's scheme. At the end
each player receives minus their OWN final score, so the game is not zero-sum
even though the winner is still whoever scores lower. Other modes: "shaped"
pays every step's change in (opponent total - own total) (zero-sum, dense);
"score_diff" pays the final score difference at the end; "win_loss" pays
+1/-1/0 at the end.

Observations follow the SkyNet repo's structured schema (``skyjo_env.py``):
board tokens ``[owner, pos, visible, value_or_UNKNOWN]`` (``visible`` means
"known to the observer"), a discard token ``[top, size]``, a global token
``[deck_size, step, current_player, phase, phase_bucket, round_index]``, the
last K public action tokens ``[actor, action_type, source, target_pos,
a_value, b_value]`` and a decision token ``[decision_phase, drawn_value]``
(the drawn card is UNKNOWN unless the observer is the player to move).

Ground-truth hidden features (hand sums etc.) are available through
``hidden_features()`` and the ``info["hidden"]`` channel of ``step()``. They
are never part of an observation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

NUM_PLAYERS = 2
HAND_SIZE = 4
UNKNOWN_VALUE = -99
ACTION_HISTORY_UNKNOWN = -127

KEEP_ACTION = 0
PEEK_ACTION = 1
TRADE_BASE = 2  # TRADE(j) = TRADE_BASE + j, j = opponent position
CALL_ACTION = TRADE_BASE + HAND_SIZE
ACTION_SPACE_SIZE = CALL_ACTION + 1

TELL_LOW = 0
TELL_HIGH = 1


class Phase(IntEnum):
    PLAY = 0
    FINAL_TURN = 1
    GAME_OVER = 2


class PublicActionType(IntEnum):
    KEEP = 1
    PEEK = 2
    TRADE = 3
    CALL = 4


@dataclass(frozen=True)
class HiddenHandConfig:
    card_values: tuple[int, ...] = tuple(range(10))
    copies_per_value: int = 4
    initial_known_cards: int = 1
    # Values of the draw pile (None = draw from the rest of the dealt deck). With only high
    # cards here, KEEP rarely helps and low cards have to be taken from the opponent by TRADE.
    draw_pile_values: tuple[int, ...] | None = (5, 6, 7, 8, 9)
    high_threshold: int = 5
    tell_accuracy: float = 0.75
    enable_trade: bool = True
    max_turns_per_player: int = 8
    min_turns_before_call: int = 2
    # If True the opponent gets one final turn after a CALL (Skyjo-style);
    # if False a CALL ends the game immediately.
    final_turn_after_call: bool = False
    caller_penalty_multiplier: int = 2
    # "own_score":  terminal reward = -(own final score) / reward_scale (not zero-sum, as in Skyjo).
    # "shaped":     every step pays the change in (opponent total - own total).
    # "score_diff": terminal reward = (opponent score - own score) / reward_scale.
    # "win_loss":   terminal reward = +1 / -1 / 0.
    reward_mode: str = "own_score"
    reward_scale: float = 1.0
    history_window_k: int = 20

    def __post_init__(self) -> None:
        if self.reward_mode not in ("shaped", "score_diff", "win_loss", "own_score"):
            raise ValueError(f"Unknown reward_mode {self.reward_mode!r}")
        if not 0.5 <= self.tell_accuracy <= 1.0:
            raise ValueError("tell_accuracy must be in [0.5, 1.0]")
        if not 0 <= self.initial_known_cards <= HAND_SIZE:
            raise ValueError(f"initial_known_cards must be in [0, {HAND_SIZE}]")
        needed = NUM_PLAYERS * (HAND_SIZE + self.max_turns_per_player)
        pile = self.draw_pile_values
        if pile is not None and len(pile) * self.copies_per_value < NUM_PLAYERS * self.max_turns_per_player:
            raise ValueError("draw_pile_values gives too few cards for these settings.")
        if len(self.card_values) * self.copies_per_value < needed:
            raise ValueError(f"The deck needs at least {needed} cards for these settings.")


@dataclass(frozen=True)
class PublicAction:
    step_index: int
    actor: int
    action_type: PublicActionType
    target_pos: int
    tell: int  # TELL_LOW / TELL_HIGH for the card looked at, else ACTION_HISTORY_UNKNOWN
    discarded: int  # card that went face-up on the discard pile, else ACTION_HISTORY_UNKNOWN
    own_pos: int = 0  # TRADE only: the actor's position that was given away

    def to_token(self) -> list[int]:
        # [actor, action_type, source, target_pos, a_value, b_value]. "source" holds the
        # actor's own position for a TRADE (target_pos is then the opponent's position).
        return [self.actor, int(self.action_type), self.own_pos, self.target_pos, self.tell, self.discarded]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_index,
            "actor": self.actor,
            "type": self.action_type.name,
            "position": self.target_pos,
            **({"own_position": self.own_pos} if self.action_type == PublicActionType.TRADE else {}),
            "tell": {TELL_LOW: "LOW", TELL_HIGH: "HIGH"}.get(self.tell),
            "discarded": None if self.discarded == ACTION_HISTORY_UNKNOWN else self.discarded,
        }


@dataclass
class Hand:
    cards: list[int]
    # known_by[pos][player]: does ``player`` know the card at ``pos``?
    known_by: list[list[bool]]

    def total(self) -> int:
        return sum(self.cards)


@dataclass
class _State:
    deck: list[int] = field(default_factory=list)
    discard_pile: list[int] = field(default_factory=list)
    hands: list[Hand] = field(default_factory=list)
    drawn: int | None = None  # the current player's privately drawn card
    current_player: int = 0
    turns_taken: list[int] = field(default_factory=lambda: [0] * NUM_PLAYERS)
    step: int = 0
    phase: Phase = Phase.PLAY
    caller: int = -1
    history: list[PublicAction] = field(default_factory=list)
    final_scores: list[int] | None = None
    winner: int = -1  # -1 = draw or not finished


def action_name(action_id: int) -> str:
    if action_id == KEEP_ACTION:
        return "KEEP"
    if action_id == PEEK_ACTION:
        return "PEEK"
    if TRADE_BASE <= action_id < CALL_ACTION:
        return f"TRADE({action_id - TRADE_BASE})"
    if action_id == CALL_ACTION:
        return "CALL"
    return f"INVALID({action_id})"


class HiddenHandEnv:
    """Two-player Hidden Hand environment with a Gym-style ``step`` like the SkyNet repo."""

    num_players = NUM_PLAYERS
    action_space_size = ACTION_SPACE_SIZE

    def __init__(self, seed: int = 0, config: HiddenHandConfig | None = None) -> None:
        self.config = config or HiddenHandConfig()
        self.initial_seed = seed
        self.rng = random.Random(seed)
        self.state = _State()
        self.reset(seed)

    # ------------------------------------------------------------------
    # Properties mirroring the repo's environments
    # ------------------------------------------------------------------
    @property
    def current_player(self) -> int:
        return self.state.current_player

    @property
    def game_over(self) -> bool:
        return self.state.phase == Phase.GAME_OVER

    @property
    def scores(self) -> list[int]:
        """Final scores once the game is over, else zeros (running totals are hidden)."""
        if self.state.final_scores is None:
            return [0] * NUM_PLAYERS
        return list(self.state.final_scores)

    @property
    def winner(self) -> int:
        """Winning player id, or -1 for a draw / unfinished game."""
        return self.state.winner

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.initial_seed = seed
        self.rng = random.Random(self.initial_seed)
        cfg = self.config

        deck = [v for v in cfg.card_values for _ in range(cfg.copies_per_value)]
        self.rng.shuffle(deck)
        self.state = _State(deck=deck)
        for owner in range(NUM_PLAYERS):
            cards = [deck.pop() for _ in range(HAND_SIZE)]
            known_by = [[False] * NUM_PLAYERS for _ in range(HAND_SIZE)]
            self.state.hands.append(Hand(cards=cards, known_by=known_by))
            for pos in sorted(self.rng.sample(range(HAND_SIZE), cfg.initial_known_cards)):
                self._look(owner, pos, step_index=0, action_type=PublicActionType.PEEK)
        if cfg.draw_pile_values is not None:
            pile = [v for v in cfg.draw_pile_values for _ in range(cfg.copies_per_value)]
            self.rng.shuffle(pile)
            self.state.deck[:] = pile
        self.state.current_player = self.rng.randrange(NUM_PLAYERS)
        self._draw()
        return self.observe(self.current_player)

    def legal_actions(self) -> list[int]:
        s = self.state
        if s.phase == Phase.GAME_OVER:
            return []
        me = s.current_player
        hand = s.hands[me]
        legal = []
        if s.drawn is not None:
            legal.append(KEEP_ACTION)
        if not all(hand.known_by[i][me] for i in range(HAND_SIZE)):
            legal.append(PEEK_ACTION)
        if self.config.enable_trade:
            legal.extend(TRADE_BASE + i for i in range(HAND_SIZE))
        if s.phase == Phase.PLAY and s.turns_taken[me] >= self.config.min_turns_before_call:
            legal.append(CALL_ACTION)
        return legal

    def legal_action_mask(self) -> list[int]:
        mask = [0] * ACTION_SPACE_SIZE
        for a in self.legal_actions():
            mask[a] = 1
        return mask

    def step(self, action_id: int) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]]:
        s = self.state
        if s.phase == Phase.GAME_OVER:
            raise RuntimeError("Game is already over. Call reset() to start a new game.")
        if action_id not in self.legal_actions():
            raise ValueError(
                f"Illegal action {action_name(action_id)} for player {s.current_player} in phase {s.phase.name}"
            )

        actor = s.current_player
        other = 1 - actor
        hand = s.hands[actor]
        totals_before = [h.total() for h in s.hands]
        drawn, s.drawn = s.drawn, None
        s.step += 1

        if action_id == KEEP_ACTION:
            pos = self.worst_slot(actor)
            old = hand.cards[pos]
            hand.cards[pos] = drawn
            hand.known_by[pos] = [False] * NUM_PLAYERS
            s.discard_pile.append(old)
            self._look(actor, pos, s.step, PublicActionType.KEEP, discarded=old)
        else:
            discarded = ACTION_HISTORY_UNKNOWN
            if drawn is not None:
                s.discard_pile.append(drawn)
                discarded = drawn
            if action_id == PEEK_ACTION:
                pos = next(i for i in range(HAND_SIZE) if not hand.known_by[i][actor])
                self._look(actor, pos, s.step, PublicActionType.PEEK, discarded=discarded)
            elif action_id < CALL_ACTION:
                mine, theirs_pos = self.worst_slot(actor), action_id - TRADE_BASE
                theirs = s.hands[other]
                hand.cards[mine], theirs.cards[theirs_pos] = theirs.cards[theirs_pos], hand.cards[mine]
                hand.known_by[mine], theirs.known_by[theirs_pos] = theirs.known_by[theirs_pos], hand.known_by[mine]
                s.history.append(
                    PublicAction(s.step, actor, PublicActionType.TRADE, theirs_pos, ACTION_HISTORY_UNKNOWN, discarded,
                                 own_pos=mine)
                )
            else:
                s.caller = actor
                s.history.append(PublicAction(s.step, actor, PublicActionType.CALL, 0, ACTION_HISTORY_UNKNOWN, discarded))

        s.turns_taken[actor] += 1
        self._advance(actor, called_now=(action_id == CALL_ACTION))

        terminated = s.phase == Phase.GAME_OVER
        rewards = self._rewards(totals_before, terminated)

        info = {
            "actor": actor,
            "action_id": action_id,
            "action_name": action_name(action_id),
            "phase": s.phase.name,
            "legal_actions_next": self.legal_actions(),
            "scores": self.scores,
            "winner": s.winner,
            "caller": s.caller,
            "hidden": self.hidden_features(actor),
        }
        return self.observe(self.current_player), rewards, terminated, info

    def worst_slot(self, player: int) -> int:
        """Position KEEP / TRADE act on: highest known card, unknown cards counting as the deck mean."""
        hand = self.state.hands[player]
        mean = sum(self.config.card_values) / len(self.config.card_values)

        def badness(pos: int) -> tuple[float, int, int]:
            known = hand.known_by[pos][player]
            return (hand.cards[pos] if known else mean, int(not known), -pos)

        return max(range(HAND_SIZE), key=badness)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observe(self, observer_player: int, history_k: int | None = None) -> dict[str, Any]:
        s = self.state
        k = self.config.history_window_k if history_k is None else history_k
        owner_order = [observer_player] + [p for p in range(NUM_PLAYERS) if p != observer_player]

        board_tokens: list[list[int]] = []
        board_views: dict[int, dict[str, list[int]]] = {}
        for owner in owner_order:
            hand = s.hands[owner]
            values: list[int] = []
            visible: list[int] = []
            for pos in range(HAND_SIZE):
                is_known = hand.known_by[pos][observer_player]
                value = hand.cards[pos] if is_known else UNKNOWN_VALUE
                values.append(value)
                visible.append(int(is_known))
                board_tokens.append([owner, pos, int(is_known), value])
            board_views[owner] = {"values": values, "visible_mask": visible}

        sees_draw = observer_player == s.current_player and s.drawn is not None
        drawn_value = s.drawn if sees_draw else UNKNOWN_VALUE
        discard_top = s.discard_pile[-1] if s.discard_pile else UNKNOWN_VALUE
        max_steps = NUM_PLAYERS * self.config.max_turns_per_player
        progress = s.step / max(1, max_steps)
        phase_bucket = 0 if progress < 0.34 else (1 if progress < 0.67 else 2)
        global_token = [len(s.deck), s.step, s.current_player, int(s.phase), phase_bucket, 0]

        last_k = s.history[-k:] if k > 0 else []
        return {
            "observer": observer_player,
            "current_player": s.current_player,
            "num_players": NUM_PLAYERS,
            "phase": s.phase.name,
            "step": s.step,
            "caller": s.caller,
            "turns_taken": list(s.turns_taken),
            "deck_size": len(s.deck),
            "drawn_card": None if drawn_value == UNKNOWN_VALUE else drawn_value,
            "own_worst_slot": self.worst_slot(observer_player),
            "discard_top": discard_top,
            "discard_pile": list(s.discard_pile),
            "board_views": board_views,
            "history": [a.to_dict() for a in last_k],
            "tokens": {
                "board_tokens": board_tokens,
                "discard_token": [discard_top, len(s.discard_pile)],
                "global_token": global_token,
                "action_tokens": [a.to_token() for a in last_k],
                "decision_token": [int(sees_draw), drawn_value],
            },
        }

    # ------------------------------------------------------------------
    # Ground truth (debug / probe channel; never part of an observation)
    # ------------------------------------------------------------------
    def hidden_features(self, observer: int) -> dict[str, float]:
        """Scalar features of the true state from ``observer``'s point of view."""
        s = self.state
        own = s.hands[observer]
        opp = s.hands[1 - observer]
        own_known = [c for c, kb in zip(own.cards, own.known_by) if kb[observer]]
        own_unknown = [c for c, kb in zip(own.cards, own.known_by) if not kb[observer]]
        opp_known = [c for c, kb in zip(opp.cards, opp.known_by) if kb[observer]]
        opp_unknown = [c for c, kb in zip(opp.cards, opp.known_by) if not kb[observer]]
        prior_mean = sum(self.config.card_values) / len(self.config.card_values)
        return {
            # Hidden from the observer. "*_excess" = sum of unknown cards minus the
            # prior expectation for that many cards, so it carries no card-count signal.
            "opp_hidden_sum": float(sum(opp_unknown)),
            "opp_hidden_excess": float(sum(opp_unknown) - prior_mean * len(opp_unknown)),
            "own_hidden_sum": float(sum(own_unknown)),
            "own_hidden_excess": float(sum(own_unknown) - prior_mean * len(own_unknown)),
            "opp_total": float(opp.total()),
            "own_total": float(own.total()),
            "total_diff": float(opp.total() - own.total()),
            # Observable to the observer (sanity controls for probes).
            "own_known_sum": float(sum(own_known)),
            "own_known_count": float(len(own_known)),
            "opp_known_sum": float(sum(opp_known)),
            "opp_known_count": float(len(opp_known)),
            "deck_size": float(len(s.deck)),
            "step": float(s.step),
        }

    def get_full_state(self) -> dict[str, Any]:
        s = self.state
        return {
            "seed": self.initial_seed,
            "deck_order_remaining": list(s.deck),
            "drawn": s.drawn,
            "discard_pile": list(s.discard_pile),
            "hands": [{"cards": list(h.cards), "known_by": [list(k) for k in h.known_by]} for h in s.hands],
            "current_player": s.current_player,
            "turns_taken": list(s.turns_taken),
            "step": s.step,
            "phase": s.phase.name,
            "caller": s.caller,
            "history": [a.to_dict() for a in s.history],
            "final_scores": None if s.final_scores is None else list(s.final_scores),
            "winner": s.winner,
        }

    def render(self, observer: int | None = None) -> str:
        """Human-readable table. With ``observer`` set, shows only what that player sees."""
        s = self.state
        show_draw = s.drawn is not None and observer in (None, s.current_player)
        lines = [
            f"step={s.step} phase={s.phase.name} to_move=P{s.current_player} "
            f"drawn={s.drawn if show_draw else '?'} deck={len(s.deck)} "
            f"discard_top={s.discard_pile[-1] if s.discard_pile else '-'} caller={s.caller}"
        ]
        for p, hand in enumerate(s.hands):
            cells = []
            for c, kb in zip(hand.cards, hand.known_by):
                if observer is None:
                    marks = "".join(str(q) for q in range(NUM_PLAYERS) if kb[q])
                    cells.append(f"{c}({marks or '-'})")
                else:
                    cells.append(str(c) if kb[observer] else "?")
            lines.append(f"  P{p}: [{' '.join(cells)}]")
        if s.history:
            recent = ", ".join(f"P{a.actor}:{a.action_type.name}({a.target_pos})" for a in s.history[-4:])
            lines.append(f"  last: {recent}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _draw(self) -> None:
        s = self.state
        s.drawn = s.deck.pop() if s.deck else None

    def _look(
        self,
        player: int,
        pos: int,
        step_index: int,
        action_type: PublicActionType,
        discarded: int = ACTION_HISTORY_UNKNOWN,
    ) -> None:
        """``player`` looks at their own card ``pos``; the table sees a noisy tell."""
        hand = self.state.hands[player]
        hand.known_by[pos][player] = True
        tell = self._tell(hand.cards[pos])
        self.state.history.append(PublicAction(step_index, player, action_type, pos, tell, discarded))

    def _tell(self, card: int) -> int:
        truth = TELL_HIGH if card >= self.config.high_threshold else TELL_LOW
        if self.rng.random() < self.config.tell_accuracy:
            return truth
        return 1 - truth

    def _advance(self, actor: int, called_now: bool) -> None:
        s = self.state
        cfg = self.config
        if s.phase == Phase.FINAL_TURN or (called_now and not cfg.final_turn_after_call):
            self._finish()
            return
        if called_now:
            s.phase = Phase.FINAL_TURN
        elif all(t >= cfg.max_turns_per_player for t in s.turns_taken):
            self._finish()
            return
        s.current_player = 1 - actor
        self._draw()

    def _finish(self) -> None:
        s = self.state
        totals = [h.total() for h in s.hands]
        scores = list(totals)
        if s.caller >= 0:
            other = 1 - s.caller
            if not totals[s.caller] < totals[other]:
                scores[s.caller] *= self.config.caller_penalty_multiplier
        s.final_scores = scores
        if scores[0] < scores[1]:
            s.winner = 0
        elif scores[1] < scores[0]:
            s.winner = 1
        else:
            s.winner = -1
        s.drawn = None
        s.phase = Phase.GAME_OVER

    def _rewards(self, totals_before: list[int], terminated: bool) -> dict[str, float]:
        s = self.state
        cfg = self.config
        rewards = {f"player_{p}": 0.0 for p in range(NUM_PLAYERS)}
        if cfg.reward_mode == "shaped":
            after = s.final_scores if terminated else [h.total() for h in s.hands]
            for p in range(NUM_PLAYERS):
                change = (after[1 - p] - after[p]) - (totals_before[1 - p] - totals_before[p])
                rewards[f"player_{p}"] = change / cfg.reward_scale
        elif terminated:
            for p in range(NUM_PLAYERS):
                if cfg.reward_mode == "win_loss":
                    r = 0.0 if s.winner == -1 else (1.0 if s.winner == p else -1.0)
                elif cfg.reward_mode == "own_score":
                    r = -s.final_scores[p] / cfg.reward_scale
                else:
                    r = (s.final_scores[1 - p] - s.final_scores[p]) / cfg.reward_scale
                rewards[f"player_{p}"] = float(r)
        return rewards
