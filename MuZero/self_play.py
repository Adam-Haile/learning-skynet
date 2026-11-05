from hidden_hand import HiddenHandEnv

from MuZero.encoder import encode_observation
from MuZero.mcts import run_mcts

def play_game(model, seed=10):

    # Create and reset the environment
    env = HiddenHandEnv(seed=seed)
    observation = env.reset()

    # Store the experience from each turn
    game_history = []

    # Keep playing until the game ends
    while not env.game_over:

        # Get the player making the current decision
        player = env.current_player

        # Convert the observation into a tensor
        encoded_observation = encode_observation(observation)

        # Find the available actions
        legal_actions = env.legal_actions()

        # Use MCTS to select an action
        action, root = run_mcts(model=model, encoded_observation=encoded_observation, legal_actions=legal_actions, current_player=player, num_simulations=16)

        # Record the search results for training
        visit_counts = [0] * 7

        for possible_action in root.children:
            child = root.children[possible_action]
            visit_counts[possible_action] = child.visit_count

        # Take the selected action
        next_observation, rewards, done, info = env.step(action)

        # Save this turn's experience
        game_history.append({
            "observation": encoded_observation,
            "player": player,
            "action": action,
            "visit_counts": visit_counts,
            "rewards": rewards
        })

        # Continue from the new observation
        observation = next_observation

    return game_history