import torch

from MuZero.networks import MuZeroNetwork
from MuZero.self_play import play_game
from MuZero.replay_buffer import ReplayBuffer
from MuZero.trainer import train_unrolled_step


def train_muzero(num_iterations=10, games_per_iteration=5, training_steps=20):

    # Create the neural networks
    model = MuZeroNetwork()

    # Create an optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Create the replay buffer
    buffer = ReplayBuffer(max_games=100)

    # Store the different training losses
    policy_history = []
    value_history = []
    reward_history = []

    # Repeat self-play and training
    for iteration in range(num_iterations):

        print("Iteration:", iteration + 1)

        # STEP 1: Play games using the current neural networks
        for game_number in range(games_per_iteration):

            seed = iteration * games_per_iteration + game_number

            game_history = play_game(model, seed=seed)

            buffer.add_game(game_history)

        print("Games stored:", len(buffer))

        # STEP 2: Train using stored experiences
        total_loss = 0.0

        for step in range(training_steps):

            # Perform one training step and get the individual losses
            losses = train_unrolled_step(
                model,
                buffer,
                optimizer,
                unroll_steps=3,
                return_details=True
            )

            # Record the total loss
            total_loss += losses["total"]

            # Record the individual losses
            policy_history.append(losses["policy"])
            value_history.append(losses["value"])
            reward_history.append(losses["reward"])

        # Calculate the average loss for this iteration
        average_loss = total_loss / training_steps

        print("Average training loss:", average_loss)
        print("-----------------------------")

    # Return the trained model and loss histories
    return model, policy_history, value_history, reward_history