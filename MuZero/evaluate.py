import random

from hidden_hand import HiddenHandEnv, make_bot

from MuZero.encoder import encode_observation
from MuZero.mcts import run_mcts


def evaluate_model(model, num_games=2000, num_simulations=16):

    wins = 0
    losses = 0
    ties = 0

    total_score = 0

    # Put neural networks in evaluation mode
    model.eval()

    for seed in range(num_games):

        env = HiddenHandEnv(seed=seed)
        observation = env.reset()

        # Alternate which player MuZero controls
        muzero_player = seed % 2

        while not env.game_over:

            player = env.current_player
            legal_actions = env.legal_actions()

            if player == muzero_player:

                # MuZero chooses an action using MCTS
                encoded_obs = encode_observation(observation)

                action, root = run_mcts(
                    model=model,
                    encoded_observation=encoded_obs,
                    legal_actions=legal_actions,
                    current_player=player,
                    num_simulations=num_simulations,
                    observation=observation,
                    game_config=env.config
                )

            else:

                # Random opponent chooses a legal action
                action = random.choice(legal_actions)

            observation, rewards, done, info = env.step(action)

        # Check the final scores
        muzero_score = env.scores[muzero_player]
        opponent_score = env.scores[1 - muzero_player]

        total_score += muzero_score

        # Lower scores are better in Hidden Hand
        if muzero_score < opponent_score:
            wins += 1

        elif muzero_score > opponent_score:
            losses += 1

        else:
            ties += 1

    print("Games played:", num_games)
    print("MuZero wins:", wins)
    print("MuZero losses:", losses)
    print("Ties:", ties)
    print("Win rate:", wins / num_games)
    print("Average score:", total_score / num_games)

    results = {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / num_games,
        "average_score": total_score / num_games
    }

    return results


def evaluate_against_bot(model, opponent_name="random", num_games=2000, num_simulations=16):
    """Evaluate MuZero against a named Hidden Hand bot.

    Alternates MuZero's seat across games and returns win/loss/score statistics.
    """
    if num_games < 1:
        raise ValueError("num_games must be positive")

    wins = 0
    losses = 0
    ties = 0
    total_score = 0

    was_training = model.training
    model.eval()

    try:
        for seed in range(num_games):
            env = HiddenHandEnv(seed=seed)
            observation = env.reset()
            muzero_player = seed % 2
            opponent = make_bot(opponent_name, seed=seed)

            while not env.game_over:
                legal_actions = env.legal_actions()

                if env.current_player == muzero_player:
                    encoded_observation = encode_observation(observation)
                    action, _root = run_mcts(
                        model=model,
                        encoded_observation=encoded_observation,
                        legal_actions=legal_actions,
                        current_player=env.current_player,
                        num_simulations=num_simulations,
                        observation=observation,
                        game_config=env.config,
                    )
                else:
                    action = opponent.select_action(observation, legal_actions)

                observation, _rewards, _done, _info = env.step(action)

            muzero_score = env.scores[muzero_player]
            opponent_score = env.scores[1 - muzero_player]
            total_score += muzero_score

            if muzero_score < opponent_score:
                wins += 1
            elif muzero_score > opponent_score:
                losses += 1
            else:
                ties += 1
    finally:
        model.train(was_training)

    return {
        "opponent": opponent_name,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / num_games,
        "average_score": total_score / num_games,
    }
