"""Turn a ``HiddenHandEnv`` observation into fixed-size token features for the network.

Token layout (``NUM_TOKENS`` rows of ``FEATURE_DIM`` features, plus a padding mask):

    8  board tokens    observer's four cards, then the opponent's four
    1  discard token
    1  global token
    1  decision token  the drawn card (only the player to move sees it)
    20 history tokens  the public action history, oldest first, padded at the end

Everything is expressed relative to the observer ("own" / "opponent"), so the same weights serve
both seats. Only ``obs["tokens"]`` is used, which holds nothing hidden.
"""
from __future__ import annotations

import numpy as np

from hidden_hand.env import ACTION_HISTORY_UNKNOWN, HAND_SIZE, UNKNOWN_VALUE

NUM_VALUES = 10
HISTORY_LEN = 20
NUM_BOARD = 2 * HAND_SIZE
NUM_TOKENS = NUM_BOARD + 3 + HISTORY_LEN
FEATURE_DIM = 28

# token type ids (one embedding each in the model)
TYPE_BOARD, TYPE_DISCARD, TYPE_GLOBAL, TYPE_DECISION, TYPE_HISTORY = range(5)
NUM_TYPES = 5
TOKEN_TYPES = np.array(
    [TYPE_BOARD] * NUM_BOARD + [TYPE_DISCARD, TYPE_GLOBAL, TYPE_DECISION] + [TYPE_HISTORY] * HISTORY_LEN,
    dtype=np.int64,
)
# slot index inside its type (board position 0..7, history position 0..19)
TOKEN_SLOTS = np.array(
    list(range(NUM_BOARD)) + [0, 0, 0] + list(range(HISTORY_LEN)),
    dtype=np.int64,
)


def _one_hot(index: int, size: int, out: np.ndarray, offset: int) -> None:
    if 0 <= index < size:
        out[offset + index] = 1.0


def encode_observation(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features [NUM_TOKENS, FEATURE_DIM] float32, pad_mask [NUM_TOKENS] bool)``.

    ``pad_mask`` is True where the token is padding and must be ignored.
    """
    tokens = obs["tokens"]
    observer = obs["observer"]
    feats = np.zeros((NUM_TOKENS, FEATURE_DIM), dtype=np.float32)
    pad = np.zeros(NUM_TOKENS, dtype=bool)

    # board: [own, pos x4, visible, value x10]
    for i, (owner, pos, visible, value) in enumerate(tokens["board_tokens"]):
        row = feats[i]
        row[0] = float(owner == observer)
        _one_hot(pos, HAND_SIZE, row, 1)
        row[5] = float(visible)
        if visible and value != UNKNOWN_VALUE:
            _one_hot(value, NUM_VALUES, row, 6)

    # discard: [top x10, top_is_none, pile_size / 16]
    top, size = tokens["discard_token"]
    row = feats[NUM_BOARD]
    if top == UNKNOWN_VALUE:
        row[10] = 1.0
    else:
        _one_hot(top, NUM_VALUES, row, 0)
    row[11] = size / 16.0

    # global: [deck / 20, step / 16, my turn, phase x3, own turns / 8, opponent turns / 8]
    deck_size, step, current, phase, _bucket, _round = tokens["global_token"]
    turns = obs["turns_taken"]
    row = feats[NUM_BOARD + 1]
    row[0] = deck_size / 20.0
    row[1] = step / 16.0
    row[2] = float(current == observer)
    _one_hot(phase, 3, row, 3)
    row[6] = turns[observer] / 8.0
    row[7] = turns[1 - observer] / 8.0

    # decision: [sees_draw, drawn x10]
    sees_draw, drawn = tokens["decision_token"]
    row = feats[NUM_BOARD + 2]
    row[0] = float(sees_draw)
    if sees_draw:
        _one_hot(drawn, NUM_VALUES, row, 1)

    # history: [actor is me, type x4, source x4, target x4, tell x3 (low/high/none), discard x11]
    history = tokens["action_tokens"][-HISTORY_LEN:]
    first = NUM_BOARD + 3
    for j, (actor, a_type, source, target, tell, discarded) in enumerate(history):
        row = feats[first + j]
        row[0] = float(actor == observer)
        _one_hot(a_type - 1, 4, row, 1)
        if a_type == 3:  # TRADE: source is the actor's own slot
            _one_hot(source, HAND_SIZE, row, 5)
        if a_type != 4:  # CALL has no target
            _one_hot(target, HAND_SIZE, row, 9)
        _one_hot(tell if tell != ACTION_HISTORY_UNKNOWN else 2, 3, row, 13)
        _one_hot(discarded if discarded != ACTION_HISTORY_UNKNOWN else NUM_VALUES, NUM_VALUES + 1, row, 16)
    pad[first + len(history):] = True
    return feats, pad


def encode_batch(observations: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    encoded = [encode_observation(o) for o in observations]
    return np.stack([e[0] for e in encoded]), np.stack([e[1] for e in encoded])
