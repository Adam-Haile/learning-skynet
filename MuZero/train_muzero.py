"""Command-line entry point: python -m MuZero.train_muzero."""

import argparse
from pathlib import Path

import torch

from MuZero.train import train_muzero


def main():
    parser = argparse.ArgumentParser(description="Train MuZero on Rosie")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--games-per-iteration", type=int, default=256)
    parser.add_argument("--training-steps", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-simulations", type=int, default=16)
    parser.add_argument("--validation-games", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="MuZero/checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable despite requesting a GPU")

    if args.smoke_test:
        if args.resume:
            parser.error("Do not resume a full run with --smoke-test")
        args.iterations = 2
        args.games_per_iteration = 2
        args.training_steps = 2
        args.batch_size = 2
        args.num_simulations = 2
        args.validation_games = 2
        args.checkpoint_interval = 1
        args.checkpoint_dir = "MuZero/smoke_checkpoints"

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"Configuration: {vars(args)}", flush=True)

    train_muzero(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        training_steps=args.training_steps,
        num_validation_games=args.validation_games,
        use_curriculum=not args.no_curriculum,
        batch_size=args.batch_size,
        num_simulations=args.num_simulations,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
