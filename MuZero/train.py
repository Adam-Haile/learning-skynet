import torch

from MuZero.networks import MuZeroNetwork
from MuZero.self_play import play_game
from MuZero.replay_buffer import ReplayBuffer
from MuZero.trainer import train_unrolled_step, validate_model


def train_muzero(
    num_iterations=10,
    games_per_iteration=5,
    training_steps=20,
    num_validation_games=10
):
    """Run self-play, training, and validation."""

    if num_iterations < 1 or games_per_iteration < 1 or training_steps < 1:
        raise ValueError("Iterations, games, and training steps must be positive")

    if num_validation_games < 1:
        raise ValueError("num_validation_games must be positive")

    # Create model and optimizer
    model = MuZeroNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0003
    )

    # Store up to 1000 training games
    buffer = ReplayBuffer(max_games=2000)

    # STEP 1: Create validation games once
    # These games are NOT added to the training replay buffer
    validation_games = []

    print("Collecting validation games...")

    for seed in range(100000, 100000 + num_validation_games):

        game_history = play_game(model, seed=seed)

        validation_games.append(game_history)

    # STEP 2: Set up loss histories
    policy_history = []
    value_history = []
    reward_history = []
    validation_history = []

    # STEP 3: Start training
    for iteration in range(num_iterations):

        print("Iteration:", iteration + 1)

        # Collect new training games
        for game_number in range(games_per_iteration):

            seed = iteration * games_per_iteration + game_number

            game_history = play_game(model, seed=seed)

            buffer.add_game(game_history)

        print("Games stored:", len(buffer))

        # Train the model
        total_loss = 0.0

        for step in range(training_steps):

            losses = train_unrolled_step(
                model,
                buffer,
                optimizer,
                unroll_steps=3,
                return_details=True
            )

            total_loss += losses["total"]

            policy_history.append(losses["policy"])
            value_history.append(losses["value"])
            reward_history.append(losses["reward"])

        # STEP 4: Measure validation loss
        validation = validate_model(
            model,
            validation_games,
            unroll_steps=3
        )

        # Store which training step this measurement belongs to
        validation["step"] = (iteration + 1) * training_steps

        validation_history.append(validation)

        average_loss = total_loss / training_steps

        print("Average training loss:", average_loss)
        print("Validation loss:", validation["total"])
        print("-----------------------------")

    # Return the trained model and all loss histories
    return (
        model,
        policy_history,
        value_history,
        reward_history,
        validation_history
    )