import random

from hidden_hand import HiddenHandEnv
from hidden_hand.bots import make_bot

from MuZero.encoder import encode_observation
from MuZero.mcts import run_mcts

def select_action_with_temperature(
    visit_counts,
    temperature=1.0
):
    """
    Sample an action using MCTS visit counts.

    temperature = 0:
        Always select the most-visited action.

    temperature = 1:
        Sample proportionally to visit counts.

    Higher temperatures:
        Produce more exploratory action selection.
    """

    # Deterministic action selection
    if temperature <= 0:

        return max(
            range(len(visit_counts)),
            key=lambda action: visit_counts[action]
        )

    # Convert visit counts into sampling weights
    weights = [
        count ** (1.0 / temperature)
        for count in visit_counts
    ]

    # Ensure MCTS produced valid visit counts
    if sum(weights) <= 0:

        raise ValueError(
            "Cannot sample action: all MCTS visit counts are zero"
        )

    # Sample an action
    selected_action = random.choices(
        population=range(len(visit_counts)),
        weights=weights,
        k=1
    )[0]

    return selected_action

def play_game(
    model,
    seed=10,
    opponent_name=None,
    explore=False
):
    """
    Generate a game for MuZero training.

    If opponent_name is None:
        MuZero controls both players.

    Otherwise:
        MuZero controls one player.
        The selected bot controls the other.

    Bot turns are recorded so that Dynamics and Value
    can still learn from those transitions.
    """

    # STEP 1: Create the environment
    env = HiddenHandEnv(seed=seed)
    observation = env.reset()

    game_history = []

    # STEP 2: Determine who controls each player
    if opponent_name is None:

        opponent = None
        muzero_player = None

    else:

        opponent = make_bot(
            opponent_name,
            seed=seed + 10000
        )

        # Alternate which seat MuZero controls
        muzero_player = seed % 2

    # STEP 3: Play the game
    while not env.game_over:

        player = env.current_player

        encoded_observation = encode_observation(
            observation
        )

        legal_actions = env.legal_actions()

        # Determine whether MuZero controls this turn
        muzero_turn = (
            opponent is None
            or player == muzero_player
        )

        if muzero_turn:

            # STEP 1: Run MCTS
            action, root = run_mcts(
                model=model,
                encoded_observation=encoded_observation,
                legal_actions=legal_actions,
                current_player=player,
                num_simulations=64,
                observation=observation,
                game_config=env.config,
                add_root_noise=explore,
                dirichlet_alpha=0.3,
                exploration_fraction=0.25
            )

            # STEP 2: Collect MCTS visit counts
            visit_counts = [0] * 7

            for possible_action, child in root.children.items():
                visit_counts[possible_action] = child.visit_count

            # STEP 3: Verify MCTS produced visits
            total_visits = sum(visit_counts)

            if total_visits == 0:
                print("WARNING: MCTS returned zero visits")
                print("Legal actions:", legal_actions)

                print(
                    "Root children:",
                    {
                        action_id: child.visit_count
                        for action_id, child in root.children.items()
                    }
                )

                raise RuntimeError(
                    "MCTS did not visit any root actions"
                )

            # STEP 4: Sample the training action
            if explore:

                game_step = len(game_history)

                if game_step < 8:
                    temperature = 1.0

                else:
                    temperature = 0.5

                action = select_action_with_temperature(
                    visit_counts,
                    temperature=temperature
                )

            # STEP 5: Mark this as an MCTS-controlled turn
            policy_valid = True

        else:

            # STEP 4B: The opponent chooses an action
            action = opponent.select_action(
                observation,
                legal_actions
            )

            # Bot actions do not have MCTS targets
            visit_counts = [0] * 7

            policy_valid = False

        # STEP 5: Execute the selected action
        next_observation, rewards, done, info = env.step(
            action
        )

        # STEP 6: Record the experience
        game_history.append({
            "observation": encoded_observation,
            "player": player,
            "action": action,
            "visit_counts": visit_counts,
            "policy_valid": policy_valid,
            "rewards": rewards
        })

        observation = next_observation

    return game_history