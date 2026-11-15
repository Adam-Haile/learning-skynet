import random
import torch
import torch.nn.functional as F
from MuZero.networks import (
    VALUE_MIN,
    VALUE_MAX,
    VALUE_SUPPORT_SIZE,
    transform_value
)

def rewards_as_list(rewards):
    """Convert Hidden Hand's reward dictionary to [player 0, player 1]."""
    if isinstance(rewards, dict):
        return [rewards["player_0"], rewards["player_1"]]
    return [rewards[0], rewards[1]]


def future_returns(game, start, discount=1.0):
    """Total observed rewards from a turn to the end of the game."""
    totals = [0.0, 0.0]
    multiplier = 1.0

    for index in range(start, len(game)):
        rewards = rewards_as_list(game[index]["rewards"])
        totals[0] += multiplier * rewards[0]
        totals[1] += multiplier * rewards[1]
        multiplier *= discount

    return totals


def policy_target(turn, device):
    """Turn MCTS visit counts into a probability distribution."""
    counts = torch.tensor(turn["visit_counts"], dtype=torch.float32, device=device)
    if counts.sum().item() == 0:
        raise ValueError("MCTS visit counts must add up to more than zero")
    return counts / counts.sum()

def calculate_losses(model, game, start, unroll_steps=3, discount=1.0):
    """Calculate losses without changing any network weights."""

    if unroll_steps < 1:
        raise ValueError("unroll_steps must be at least 1")

    if not 0 <= start < len(game):
        raise ValueError("Starting turn is outside the game")

    device = next(model.parameters()).device

    observation = game[start]["observation"].to(device).unsqueeze(0)

    hidden_state, policy, value, value_logits = (
        model.initial_inference_with_logits(observation)
    )

    all_losses = []
    policy_losses = []
    value_losses = []
    reward_losses = []

    # Calculate loss for the starting observation
    target_policy = policy_target(game[start], device).unsqueeze(0)

    target_value = torch.tensor(
        [future_returns(game, start, discount)],
        dtype=torch.float32,
        device=device
    )

    policy_loss = -(target_policy * F.log_softmax(policy, dim=1)).sum()
    value_loss = categorical_value_loss(
        value_logits,
        target_value
    )
    policy_losses.append(policy_loss)
    value_losses.append(value_loss)

    all_losses.extend([policy_loss, value_loss])

    # Predict what happens after each recorded action
    end = min(start + unroll_steps, len(game))

    for index in range(start, end):

        turn = game[index]

        action = torch.tensor(
            [turn["action"]],
            dtype=torch.long,
            device=device
        )

        hidden_state, predicted_reward, policy, value, value_logits = (
            model.recurrent_inference_with_logits(
                hidden_state,
                action
            )
        )

        # Compare the predicted immediate reward with the real reward
        actual_reward = torch.tensor(
            [rewards_as_list(turn["rewards"])],
            dtype=torch.float32,
            device=device
        )

        reward_loss = F.smooth_l1_loss(
            predicted_reward,
            actual_reward,
            beta=5.0
        )

        reward_losses.append(reward_loss)
        all_losses.append(reward_loss)

        # Compare predictions with the next turn
        next_index = index + 1

        if next_index < len(game):

            target_policy = policy_target(game[next_index], device).unsqueeze(0)

            target_value = torch.tensor(
                [future_returns(game, next_index, discount)],
                dtype=torch.float32,
                device=device
            )

            policy_loss = -(target_policy * F.log_softmax(policy, dim=1)).sum()
            value_loss = categorical_value_loss(
                value_logits,
                target_value
            )

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

            all_losses.extend([policy_loss, value_loss])

        else:
            # No future reward after the final action
            value_loss = categorical_value_loss(
                value_logits,
                torch.zeros_like(value)
            )

            value_losses.append(value_loss)
            all_losses.append(value_loss)

            break

    # Average each loss independently
    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.stack(value_losses).mean()
    reward_loss = torch.stack(reward_losses).mean()

    # Explicitly balance the three objectives
    policy_weight = 1.0
    value_weight = 1.0
    reward_weight = 1.0

    total_loss = (
            policy_weight * policy_loss
            + value_weight * value_loss
            + reward_weight * reward_loss
    )

    return {
        "total": total_loss,
        "policy": policy_loss,
        "value": value_loss,
        "reward": reward_loss
    }


def train_unrolled_step(
    model,
    buffer,
    optimizer,
    unroll_steps=3,
    discount=1.0,
    return_details=False,
    terminal_probability=0.30
):
    """Train MuZero using a sampled game and starting turn."""

    # Select a random game from the replay buffer
    game = buffer.sample_games(1)[0]

    # Sometimes train directly on the final action
    if random.random() < terminal_probability:
        start = len(game) - 1

    else:
        # Otherwise select a random starting turn
        start = random.randrange(len(game))

    model.train()

    # Calculate policy, value, and reward losses
    losses = calculate_losses(
        model,
        game,
        start,
        unroll_steps=unroll_steps,
        discount=discount
    )

    # Update the neural networks
    optimizer.zero_grad()

    losses["total"].backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0
    )

    optimizer.step()

    # Return individual loss measurements when requested
    if return_details:
        return {
            name: loss.item()
            for name, loss in losses.items()
        }

    return losses["total"].item()

def validate_model(
    model,
    validation_games,
    unroll_steps=3,
    discount=1.0
):
    """Calculate validation losses without updating the model."""

    if len(validation_games) == 0:
        raise ValueError("Validation games cannot be empty")

    was_training = model.training

    model.eval()

    totals = {
        "total": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "reward": 0.0
    }

    samples = 0

    try:
        # No gradients or weight updates during validation
        with torch.no_grad():

            for game in validation_games:

                # Use the same two starting turns every time
                starts = sorted(set([
                    0,
                    len(game) // 2,
                    len(game) - 1
                ]))

                for start in starts:

                    losses = calculate_losses(
                        model,
                        game,
                        start,
                        unroll_steps,
                        discount
                    )

                    for name in totals:
                        totals[name] += losses[name].item()

                    samples += 1

    finally:
        # Restore the model's previous mode
        model.train(was_training)

    # Return average validation losses
    return {
        name: total / samples
        for name, total in totals.items()
    }

def categorical_value_loss(value_logits, target_returns):
    """
    Calculate categorical cross-entropy using
    compressed future returns.

    value_logits:
        [batch_size, num_players, 29]

    target_returns:
        [batch_size, num_players]
    """

    # STEP 1: Compress the original return values
    compressed_returns = transform_value(
        target_returns
    )

    # STEP 2: Make sure values fit within our support
    if torch.any(
        (compressed_returns < VALUE_MIN)
        | (compressed_returns > VALUE_MAX)
    ):
        raise ValueError(
            "Compressed return is outside "
            "the supported range [-14, 14]"
        )

    # STEP 3: Convert compressed values into bin positions
    positions = compressed_returns - VALUE_MIN

    # STEP 4: Find the two nearest categories
    lower = positions.floor().long()
    upper = positions.ceil().long()

    # STEP 5: Calculate probability weights
    upper_weight = positions - lower
    lower_weight = 1.0 - upper_weight

    # STEP 6: Create the target distribution
    target_distribution = torch.zeros_like(
        value_logits
    )

    # STEP 7: Assign probability to the lower category
    target_distribution.scatter_add_(
        dim=-1,
        index=lower.unsqueeze(-1),
        src=lower_weight.unsqueeze(-1)
    )

    # STEP 8: Assign probability to the upper category
    target_distribution.scatter_add_(
        dim=-1,
        index=upper.unsqueeze(-1),
        src=upper_weight.unsqueeze(-1)
    )

    # STEP 9: Convert predicted logits into log probabilities
    log_probabilities = F.log_softmax(
        value_logits,
        dim=-1
    )

    # STEP 10: Calculate categorical cross-entropy
    loss = -(
        target_distribution * log_probabilities
    ).sum(dim=-1).mean()

    return loss