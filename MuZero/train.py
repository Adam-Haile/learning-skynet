import torch
import random
import os
import signal
from pathlib import Path

from MuZero.networks import MuZeroNetwork
from MuZero.self_play import play_game
from MuZero.replay_buffer import ReplayBuffer
from MuZero.trainer import train_unrolled_step, validate_model

_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True
    print(f"Received signal {signum}; stopping after the current iteration.", flush=True)


def _save_checkpoint(path, iteration, model, optimizer, buffer,
                     validation_games, histories, config):
    """Write a complete resumable checkpoint, then atomically publish it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "buffer": buffer,
        "validation_games": validation_games,
        "histories": histories,
        "config": config,
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


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
    bot_names=("random", "greedy", "tell_reader"),
    batch_size=1,
    num_simulations=64,
    checkpoint_interval=0,
    checkpoint_dir="checkpoints",
    resume_path=None,
    replay_max_games=2000
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MuZeroNetwork().to(device)
    print(f"Training device: {device}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0003
    )

    # Store up to 2000 training games
    buffer = ReplayBuffer(max_games=replay_max_games)

    config = {
        "num_iterations": num_iterations,
        "games_per_iteration": games_per_iteration,
        "training_steps": training_steps,
        "batch_size": batch_size,
        "num_simulations": num_simulations,
        "use_curriculum": use_curriculum,
        "bot_names": tuple(bot_names),
        "replay_max_games": replay_max_games,
        "num_validation_games": num_validation_games,
    }

    start_iteration = 0
    checkpoint = None
    if resume_path is not None:
        # Only load checkpoints you created or explicitly trust.
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        for name in ("games_per_iteration", "training_steps", "batch_size",
                     "num_simulations", "use_curriculum", "bot_names",
                     "replay_max_games", "num_validation_games"):
            if checkpoint["config"][name] != config[name]:
                raise ValueError(f"Cannot resume: {name} differs from checkpoint")
        if num_iterations < checkpoint["iteration"]:
            raise ValueError("num_iterations is less than completed iterations")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        buffer = checkpoint["buffer"]
        start_iteration = checkpoint["iteration"]
        random.setstate(checkpoint["random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"Resuming from iteration {start_iteration}", flush=True)

    # STEP 2: Generate validation games
    # These are not added to the training buffer
    validation_games = [] if checkpoint is None else checkpoint["validation_games"]

    print("Collecting validation games...")

    if checkpoint is None:
        for seed in range(100000, 100000 + num_validation_games):
            game_history = play_game(model, seed=seed, num_simulations=num_simulations)
            validation_games.append(game_history)

    # Set up loss histories
    histories = ({"policy": [], "value": [], "reward": [], "validation": []}
                 if checkpoint is None else checkpoint["histories"])
    policy_history = histories["policy"]
    value_history = histories["value"]
    reward_history = histories["reward"]
    validation_history = histories["validation"]

    # Slurm can signal shortly before the job's wall-time limit.
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # STEP 3: Start training
    for iteration in range(start_iteration, num_iterations):

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
                explore=True,
                num_simulations=num_simulations
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
                return_details=True,
                batch_size=batch_size
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

        completed = iteration + 1
        if checkpoint_interval and (
            completed % checkpoint_interval == 0
            or completed == num_iterations
            or _stop_requested
        ):
            checkpoint_path = Path(checkpoint_dir) / f"iteration_{completed:04d}.pt"
            _save_checkpoint(checkpoint_path, completed, model, optimizer, buffer,
                             validation_games, histories, config)
            _save_checkpoint(Path(checkpoint_dir) / "latest.pt", completed,
                             model, optimizer, buffer, validation_games, histories, config)
            print(f"Saved checkpoint: {checkpoint_path}", flush=True)

        if _stop_requested:
            print("Stopped safely after saving checkpoint.", flush=True)
            break

    # STEP 7: Return the model and histories
    return (
        model,
        policy_history,
        value_history,
        reward_history,
        validation_history
    )
