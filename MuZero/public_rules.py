
"""Track future legal actions using public Hidden Hand information."""

from hidden_hand import HiddenHandConfig

def initial_public_state(observation, config=None):
    """Reconstruct which card positions each player knows."""
    config = config or HiddenHandConfig()
    history = observation["history"]

    initial_events = [
        event for event in history
        if event["step"] == 0
    ]

    expected = 2 * config.initial_known_cards

    if len(initial_events) != expected:
        raise ValueError(
            "Public action history is incomplete. "
            "Increase history_window_k."
        )

    # Eight card positions.
    # Each number contains two bits:
    # bit 0 = player 0 knows the card
    # bit 1 = player 1 knows the card
    knowledge = [0] * 8

    for event in history:
        actor = event["actor"]
        action_type = event["type"]
        position = event["position"]

        index = 4 * actor + position

        if action_type == "PEEK":
            knowledge[index] |= 1 << actor

        elif action_type == "KEEP":
            knowledge[index] = 1 << actor

        elif action_type == "TRADE":
            other = 1 - actor

            source = 4 * actor + event["own_position"]
            target = 4 * other + position

            knowledge[source], knowledge[target] = (
                knowledge[target],
                knowledge[source]
            )

    return {
        "player": observation["current_player"],
        "turns": tuple(observation["turns_taken"]),
        "phase": observation["phase"],
        "deck_size": observation["deck_size"],
        "has_drawn": observation["drawn_card"] is not None,
        "possible_knowledge": {tuple(knowledge)}
    }


def advance_public_state(state, action, config=None):
    """Apply an action to the public game state."""
    config = config or HiddenHandConfig()

    player = state["player"]
    other = 1 - player

    turns = list(state["turns"])
    turns[player] += 1

    # Determine the next phase
    if state["phase"] == "FINAL_TURN":
        next_phase = "GAME_OVER"

    elif action == 6:
        if config.final_turn_after_call:
            next_phase = "FINAL_TURN"
        else:
            next_phase = "GAME_OVER"

    elif all(
        count >= config.max_turns_per_player
        for count in turns
    ):
        next_phase = "GAME_OVER"

    else:
        next_phase = "PLAY"

    possibilities = set()

    for old_knowledge in state["possible_knowledge"]:

        if action == 1:
            # PEEK reveals the player's first unknown card
            knowledge = list(old_knowledge)

            for position in range(4):
                index = 4 * player + position

                if not (knowledge[index] & (1 << player)):
                    knowledge[index] |= 1 << player
                    break

            possibilities.add(tuple(knowledge))

        elif action == 0:
            # KEEP replaces the worst card.
            # The source position depends on hidden values.
            for position in range(4):
                knowledge = list(old_knowledge)
                knowledge[4 * player + position] = 1 << player

                possibilities.add(tuple(knowledge))

        elif 2 <= action <= 5:
            # TRADE exchanges cards.
            # Keep every possible source position.
            target = 4 * other + (action - 2)

            for position in range(4):
                knowledge = list(old_knowledge)
                source = 4 * player + position

                knowledge[source], knowledge[target] = (
                    knowledge[target],
                    knowledge[source]
                )

                possibilities.add(tuple(knowledge))

        else:
            # CALL doesn't change card knowledge
            possibilities.add(old_knowledge)

    # Draw a new card only if the game continues
    will_draw = (
        next_phase != "GAME_OVER"
        and state["deck_size"] > 0
    )

    return {
        "player": other,
        "turns": tuple(turns),
        "phase": next_phase,
        "deck_size": state["deck_size"] - int(will_draw),
        "has_drawn": will_draw,
        "possible_knowledge": possibilities
    }


def future_legal_actions(state, config=None):
    """Determine which actions are guaranteed legal."""
    config = config or HiddenHandConfig()

    if state["phase"] == "GAME_OVER":
        return []

    player = state["player"]
    actions = []

    # KEEP requires a drawn card
    if state["has_drawn"]:
        actions.append(0)

    # PEEK requires an unknown card.
    # Check every possible knowledge state.
    peek_is_certainly_legal = all(
        any(
            not (
                knowledge[4 * player + position]
                & (1 << player)
            )
            for position in range(4)
        )
        for knowledge in state["possible_knowledge"]
    )

    if peek_is_certainly_legal:
        actions.append(1)

    # TRADE actions
    if config.enable_trade:
        actions.extend([2, 3, 4, 5])

    # CALL becomes legal after enough turns
    if (
        state["phase"] == "PLAY"
        and state["turns"][player]
        >= config.min_turns_before_call
    ):
        actions.append(6)

    return actions
