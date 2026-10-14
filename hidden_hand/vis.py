"""
What a Hidden Hand table looks like from one seat. The game window
(``python -m hidden_hand.play --vis``, see ``window.py``) draws exactly this.

``table_view(env, view)`` lists everything there is to draw:

* ``view=None`` is the spectator view: every card face up, which players know
  each card, the drawn card, and whether each tell told the truth.
* ``view=p`` holds only what player ``p`` sees (the same information as
  ``env.observe(p)``), so a human can play from it without seeing hidden cards.

Once the game is over every card is revealed in either view.
"""
from __future__ import annotations

from typing import Any, Sequence

from hidden_hand.env import (
    ACTION_HISTORY_UNKNOWN,
    HAND_SIZE,
    NUM_PLAYERS,
    TELL_HIGH,
    HiddenHandEnv,
    PublicAction,
    PublicActionType,
)

# A described action is a list of (text, tag) parts. Tags name a style: "p0"/"p1" for the
# players, "high"/"low" for tells, "true"/"false" for whether a tell was right, "dim".
Part = tuple[str, str]


def tell_truths(env: HiddenHandEnv, truths: list[bool | None]) -> list[bool | None]:
    """
    Extend ``truths`` so it has one entry per public action: whether its tell was right
    (None for actions without a tell). Call it after ``reset()`` and after every ``step()``:
    a tell is judged against the card that is in the looked-at slot at that moment.
    """
    s = env.state
    for action in s.history[len(truths):]:
        if action.tell == ACTION_HISTORY_UNKNOWN:
            truths.append(None)
        else:
            card = s.hands[action.actor].cards[action.target_pos]
            truths.append((card >= env.config.high_threshold) == (action.tell == TELL_HIGH))
    return truths


def describe_action(action: PublicAction, names: Sequence[str] | None = None, truth: bool | None = None) -> list[Part]:
    """One public history entry, e.g. ``P0 greedy peeks at slot 2 · tell LOW · discards 7``."""
    who = f"P{action.actor}" + (f" {names[action.actor]}" if names else "")
    kind = action.action_type
    if kind == PublicActionType.PEEK and action.step_index == 0:
        what = f" looks at slot {action.target_pos} (deal)"
    elif kind == PublicActionType.PEEK:
        what = f" peeks at slot {action.target_pos}"
    elif kind == PublicActionType.KEEP:
        what = f" keeps the draw in slot {action.target_pos}"
    elif kind == PublicActionType.TRADE:
        what = f" trades slot {action.own_pos} for P{1 - action.actor}'s slot {action.target_pos}"
    else:
        what = " calls"
    parts: list[Part] = [(who, f"p{action.actor}"), (what, "")]
    if action.tell != ACTION_HISTORY_UNKNOWN:
        high = action.tell == TELL_HIGH
        parts += [(" · tell ", ""), ("HIGH" if high else "LOW", "high" if high else "low")]
        if truth is not None:
            parts.append((" ✓", "true") if truth else (" ✗ wrong", "false"))
    if action.discarded != ACTION_HISTORY_UNKNOWN:
        parts.append((f" · discards {action.discarded}", "dim"))
    return parts


def describe_plain(action: PublicAction, names: Sequence[str] | None = None) -> str:
    return "".join(text for text, _ in describe_action(action, names))


def _last_action(env: HiddenHandEnv) -> PublicAction | None:
    s = env.state
    if s.step > 0 and s.history and s.history[-1].step_index == s.step:
        return s.history[-1]
    return None


def _changed_slots(action: PublicAction | None) -> set[tuple[int, int]]:
    if action is None or action.action_type == PublicActionType.CALL:
        return set()
    if action.action_type == PublicActionType.TRADE:
        return {(action.actor, action.own_pos), (1 - action.actor, action.target_pos)}
    return {(action.actor, action.target_pos)}


def table_view(
    env: HiddenHandEnv,
    view: int | None = None,
    names: Sequence[str] = ("", ""),
    truths: Sequence[bool | None] | None = None,
) -> dict[str, Any]:
    """Everything to draw for ``view`` (None = spectator), containing nothing that seat may not see."""
    s = env.state
    reveal = view is None or env.game_over
    to_move = None if env.game_over else s.current_player
    names = [n or f"player {p}" for p, n in enumerate(names)]
    last = _last_action(env)
    changed = _changed_slots(last)

    hands = []
    for owner in range(NUM_PLAYERS):
        hand = s.hands[owner]
        worst = env.worst_slot(owner) if to_move == owner and (reveal or view == owner) else None
        cards = []
        for pos in range(HAND_SIZE):
            visible = reveal or hand.known_by[pos][view]
            cards.append({
                "value": hand.cards[pos] if visible else None,
                "seen_by": [p for p in range(NUM_PLAYERS) if hand.known_by[pos][p]] if reveal else None,
                "changed": (owner, pos) in changed,
                "worst": pos == worst,
            })
        seen = [c["value"] for c in cards if c["value"] is not None]
        hands.append({
            "owner": owner,
            "name": names[owner],
            "cards": cards,
            "turns": s.turns_taken[owner],
            "total": hand.total() if reveal else None,
            "seen_sum": sum(seen),
            "unseen": HAND_SIZE - len(seen),
        })

    history = []
    for i, action in enumerate(s.history):
        truth = truths[i] if reveal and truths is not None and i < len(truths) else None
        history.append({"step": action.step_index, "parts": describe_action(action, names, truth),
                        "newest": action is last})

    result = None
    if env.game_over:
        totals = [h.total() for h in s.hands]
        result = {"caller": s.caller, "totals": totals, "scores": list(s.final_scores), "winner": s.winner,
                  "doubled": s.caller >= 0 and s.final_scores[s.caller] != totals[s.caller]}

    has_draw = to_move is not None and s.drawn is not None
    return {
        "view": view,
        "reveal": reveal,
        "names": names,
        "step": s.step,
        "max_steps": NUM_PLAYERS * env.config.max_turns_per_player,
        "max_turns": env.config.max_turns_per_player,
        "phase": s.phase.name,
        "high_threshold": env.config.high_threshold,
        "to_move": to_move,
        "hands": hands,
        "has_draw": has_draw,
        "drawn": s.drawn if has_draw and (reveal or view == to_move) else None,
        "deck_size": len(s.deck),
        "discards": list(s.discard_pile),
        "history": history,
        "result": result,
    }
