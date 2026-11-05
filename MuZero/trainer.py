import random
import torch
import torch.nn.functional as F


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


def train_unrolled_step(model, buffer, optimizer, unroll_steps=3, discount=1.0, return_details=False):
    """Train representation, dynamics, reward and prediction networks together."""
    if unroll_steps < 1:
        raise ValueError("unroll_steps must be at least 1")

    game = buffer.sample_games(1)[0]
    start = random.randrange(len(game))
    device = next(model.parameters()).device

    # Start from an observation that was actually seen during the game.
    observation = game[start]["observation"].to(device).unsqueeze(0)
    model.train()
    hidden_state, policy, value = model.initial_inference(observation)

    losses = []
    policy_losses = []
    value_losses = []
    reward_losses = []

    # Initial state's policy and future-value predictions.
    target_policy = policy_target(game[start], device).unsqueeze(0)
    target_value = torch.tensor(
        [future_returns(game, start, discount)],
        dtype=torch.float32,
        device=device
    )
    policy_loss = -(target_policy * F.log_softmax(policy, dim=1)).sum()
    value_loss = F.mse_loss(value, target_value)
    policy_losses.append(policy_loss)
    value_losses.append(value_loss)
    losses.append(policy_loss)
    losses.append(value_loss)

    # Predict consecutive transitions using the dynamics network.
    end = min(start + unroll_steps, len(game))

    for index in range(start, end):
        turn = game[index]
        action = torch.tensor([turn["action"]], dtype=torch.long, device=device)

        hidden_state, predicted_reward, policy, value = model.recurrent_inference(
            hidden_state, action
        )

        # A reward belongs to the action/transition at this index.
        actual_reward = torch.tensor(
            [rewards_as_list(turn["rewards"])],
            dtype=torch.float32,
            device=device
        )
        reward_loss = F.mse_loss(predicted_reward, actual_reward)
        reward_losses.append(reward_loss)
        losses.append(reward_loss)

        next_index = index + 1
        if next_index < len(game):
            # The next recorded turn is the next real state in this trajectory.
            target_policy = policy_target(game[next_index], device).unsqueeze(0)
            target_value = torch.tensor(
                [future_returns(game, next_index, discount)],
                dtype=torch.float32,
                device=device
            )
            policy_loss = -(target_policy * F.log_softmax(policy, dim=1)).sum()
            value_loss = F.mse_loss(value, target_value)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            losses.append(policy_loss)
            losses.append(value_loss)
        else:
            # The trajectory ended: future return after the final action is zero.
            value_loss = F.mse_loss(value, torch.zeros_like(value))
            value_losses.append(value_loss)
            losses.append(value_loss)
            break

    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if return_details:
        return {
            "total": total_loss.item(),
            "policy": torch.stack(policy_losses).mean().item(),
            "value": torch.stack(value_losses).mean().item(),
            "reward": torch.stack(reward_losses).mean().item(),
        }

    return total_loss.item()