import torch

MAX_HISTORY = 20
ACTION_FEATURES = 6
OBSERVATION_SIZE = 303
UNKNOWN_CARD = -99
UNKNOWN_ACTION = -127

"""
Converts a Hidden Hand observation into
a fixed-size PyTorch tensor.

Returns:
    Tensor containing 303 features.
"""
def encode_observation(obs):

    tokens = obs["tokens"]

    # Convert observation tokens into tensors
    board = torch.tensor(
        tokens["board_tokens"],
        dtype=torch.float32
    )

    discard = torch.tensor(
        tokens["discard_token"],
        dtype=torch.float32
    )

    global_state = torch.tensor(
        tokens["global_token"],
        dtype=torch.float32
    )

    decision = torch.tensor(
        tokens["decision_token"],
        dtype=torch.float32
    )

    actions = torch.tensor(
        tokens["action_tokens"],
        dtype=torch.float32
    ).reshape(-1, ACTION_FEATURES)

    # 2. Remove unknown-cards, keeping the existing board visibility
    # and decision flags. Discard needs its own flag to distinguish empty from 0.
    for card in board:
        if card[3] == UNKNOWN_CARD:
            card[3] = 0.0

    discard_known = 1.0
    if discard[0] == UNKNOWN_CARD:
        discard_known = 0.0
        discard[0] = 0.0

    if decision[1] == UNKNOWN_CARD:
        decision[1] = 0.0

    # 3. Clean action-field placeholder and record which fields were supplied.
    value_mask = torch.ones_like(actions)
    for i in range(len(actions)):
        for j in range(ACTION_FEATURES):
            if actions[i][j] == UNKNOWN_ACTION:
                actions[i][j] = 0.0
                value_mask[i][j] = 0.0

    # 4. Pad action rows AND their availability masks to 20 x 6.
    padded_actions = torch.zeros(MAX_HISTORY, ACTION_FEATURES)
    padded_value_mask = torch.zeros(MAX_HISTORY, ACTION_FEATURES)
    history_mask = torch.zeros(MAX_HISTORY)

    recent_actions = actions[-MAX_HISTORY:]
    recent_value_masks = value_mask[-MAX_HISTORY:]
    for i in range(len(recent_actions)):
        padded_actions[i] = recent_actions[i]
        padded_value_mask[i] = recent_value_masks[i]
        history_mask[i] = 1.0

    # 5. Scale numerical fields using default Hidden Hand configuration.
    for card in board:
        card[1] /= 3.0  # Card position
        card[3] /= 9.0  # Known card value

    discard[0] /= 9.0
    discard[1] /= 16.0

    global_state[0] /= 20.0  # Draw-pile size
    global_state[1] /= 16.0  # Turn count
    global_state[3] /= 2.0   # Phase code
    global_state[4] /= 2.0   # Phase bucket

    decision[1] /= 9.0

    for i in range(MAX_HISTORY):
        if history_mask[i] == 0:
            continue
        action = padded_actions[i]
        action[1] /= 4.0  # Public action type
        action[2] /= 3.0  # Source card position
        action[3] /= 3.0  # Target card position
        action[5] /= 9.0  # Discarded card value

    # 6. Flatten and join all parts into the 303-dimensional input.
    encoded_observation = torch.cat([
        board.flatten(),
        discard,
        torch.tensor([discard_known], dtype=torch.float32),
        global_state,
        decision,
        padded_actions.flatten(),
        history_mask,
        padded_value_mask.flatten(),
    ])

    # Check if the observation has the correct number of features
    if len(encoded_observation) != OBSERVATION_SIZE:
        raise ValueError("Encoded observation has the wrong size!")

    return encoded_observation