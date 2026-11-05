"""Small neural networks for a Hidden Hand MuZero baseline.

The model predicts values and rewards for both players, since Hidden Hand's
own_score rewards are not zero-sum. Outputs are untrained until the RL loop
optimizes them.
"""

import torch
from torch import nn
from torch.nn import functional as F

from MuZero.encoder import OBSERVATION_SIZE

HIDDEN_SIZE = 128
ACTION_SPACE_SIZE = 7
NUM_PLAYERS = 2

class RepresentationNetwork(nn.Module):
    """Turn an encoded observation into a learned hidden state."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

    def forward(self, observation):
        return self.layers(observation)


class DynamicsNetwork(nn.Module):
    """Predict the next hidden state and immediate rewards after an action."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + ACTION_SPACE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(HIDDEN_SIZE, NUM_PLAYERS)

    def forward(self, hidden_state, action):
        # Convert action ID(s), e.g. KEEP=0, into seven binary input features.
        action = torch.as_tensor(action, device=hidden_state.device)
        action = action.long()
        if action.ndim == 0:
            action = action.unsqueeze(0)
        action_features = F.one_hot(action, num_classes=ACTION_SPACE_SIZE).float()

        # Match single-observation inputs [128] or batched inputs [B, 128].
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.unsqueeze(0)
        combined = torch.cat([hidden_state, action_features], dim=-1)

        next_hidden_state = self.layers(combined)
        predicted_rewards = self.reward_head(next_hidden_state)
        return next_hidden_state, predicted_rewards


class PredictionNetwork(nn.Module):
    """Estimate action preferences and future returns from a hidden state."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(HIDDEN_SIZE, ACTION_SPACE_SIZE)
        self.value_head = nn.Linear(HIDDEN_SIZE, NUM_PLAYERS)

    def forward(self, hidden_state):
        features = self.layers(hidden_state)
        policy_logits = self.policy_head(features)
        predicted_values = self.value_head(features)
        return policy_logits, predicted_values


class MuZeroNetwork(nn.Module):
    """Combine the three MuZero learned functions."""

    def __init__(self):
        super().__init__()
        self.representation = RepresentationNetwork()
        self.dynamics = DynamicsNetwork()
        self.prediction = PredictionNetwork()

    def initial_inference(self, observation):
        """Evaluate a real observation at the root of search."""
        hidden_state = self.representation(observation)
        policy_logits, values = self.prediction(hidden_state)
        return hidden_state, policy_logits, values

    def recurrent_inference(self, hidden_state, action):
        """Evaluate a hypothetical action in search."""
        next_hidden_state, rewards = self.dynamics(hidden_state, action)
        policy_logits, values = self.prediction(next_hidden_state)
        return next_hidden_state, rewards, policy_logits, values
