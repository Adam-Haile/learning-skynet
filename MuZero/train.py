import torch
import random

from MuZero.networks import MuZeroNetwork
from MuZero.self_play import play_game
from MuZero.replay_buffer import ReplayBuffer
from MuZero.trainer import train_unrolled_step, validate_model


def get_self_play_probability(iteration, total_iterations):
    """
    Gradually transition from bot training to self-play.

    Training stages:
        0-20%:    20% self-play
        20-40%:   40% self-play
        40-60%:   60% self-play
        60-80%:   80% self-play
        80-100%: 100% self-play
    """

    progress = iteration / total_iterations

    if progress < 0.20:
        return 0.20

    elif progress < 0.40:
        return 0.40

    elif progress < 0.60:
        return 0.60

    elif progress < 0.80:
        return 0.80

    return 1.0


def train_muzero(
    num_iterations=10,
    games_per_iteration=5,
    training_steps=20,
    num_validation_games=10,
    use_curriculum=True,
    bot_names=("random", "greedy", "tell_reader")
):
    """Run curriculum training, self-play, and validation."""

    if num_iterations < 1 or games_per_iteration < 1 or training_steps < 1:
        raise ValueError(
            "Iterations, games, and training steps must be positive"
        )

    if num_validation_games < 1:
        raise ValueError(
            "num_validation_games must be positive"
        )

    if use_curriculum and len(bot_names) == 0:
        raise ValueError(
            "At least one opponent bot must be provided"
        )

    # STEP 1: Create model and optimizer
    model = MuZeroNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0003
    )

    # Store up to 2000 training games
    buffer = ReplayBuffer(max_games=2000)

    # STEP 2: Generate validation games
    # These are not added to the training buffer
    validation_games = []

    print("Collecting validation games...")

    for seed in range(
        100000,
        100000 + num_validation_games
    ):

        game_history = play_game(
            model,
            seed=seed
        )

        validation_games.append(game_history)

    # Set up loss histories
    policy_history = []
    value_history = []
    reward_history = []
    validation_history = []

    # STEP 3: Start training
    for iteration in range(num_iterations):

        print("Iteration:", iteration + 1)

        # Determine the curriculum stage
        if use_curriculum:

            self_play_probability = get_self_play_probability(
                iteration,
                num_iterations
            )

        else:

            self_play_probability = 1.0

        # Track game types
        self_play_games = 0
        bot_games = 0

        # STEP 4: Generate training games
        for game_number in range(games_per_iteration):

            seed = (
                iteration * games_per_iteration
                + game_number
            )

            # Decide whether to use self-play
            if random.random() < self_play_probability:

                opponent_name = None

                self_play_games += 1

            else:

                opponent_name = random.choice(
                    bot_names
                )

                bot_games += 1

            # Generate the game
            game_history = play_game(
                model,
                seed=seed,
                opponent_name=opponent_name,
                explore=True
            )

            # Add the experience to the replay buffer
            buffer.add_game(game_history)

        # Display curriculum information
        print(
            "Self-play probability:",
            round(self_play_probability, 2)
        )

        print(
            "Self-play games:",
            self_play_games
        )

        print(
            "Bot games:",
            bot_games
        )

        print(
            "Games stored:",
            len(buffer)
        )

        # STEP 5: Train the model
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

            policy_history.append(
                losses["policy"]
            )

            value_history.append(
                losses["value"]
            )

            reward_history.append(
                losses["reward"]
            )

        # STEP 6: Measure validation loss
        validation = validate_model(
            model,
            validation_games,
            unroll_steps=3
        )

        # Record the training step
        validation["step"] = (
            (iteration + 1) * training_steps
        )

        validation_history.append(
            validation
        )

        # Calculate average training loss
        average_loss = (
            total_loss / training_steps
        )

        print(
            "Average training loss:",
            average_loss
        )

        print(
            "Validation loss:",
            validation["total"]
        )

        print("-----------------------------")

    # STEP 7: Return the model and histories
    return (
        model,
        policy_history,
        value_history,
        reward_history,
        validation_history
    )