"""Small neural networks for a Hidden Hand MuZero baseline.

The model predicts values and rewards for both players, since Hidden Hand's
own_score rewards are not zero-sum. Outputs are untrained until the RL loop
optimizes them.
"""

import torch
from torch import nn
from torch.nn import functional as F
from MuZero.encoder import (
    OBSERVATION_SIZE,
    MAX_HISTORY,
    ACTION_FEATURES
)

HIDDEN_SIZE = 512
ACTION_SPACE_SIZE = 7
NUM_PLAYERS = 2

# Shared scale for the dynamics rewards and prediction values.
# The observed terminal rewards were roughly -68 to -5. This is an
# experimental scale, not a guaranteed bound on the game's scores.
# Keep environment rewards and score-based evaluation in original units.
RETURN_SCALE = 70.0


def to_model_scale(returns):
    """Convert environment reward/return units to model units."""
    return returns / RETURN_SCALE


def to_environment_scale(predictions):
    """Convert model reward/value predictions back to game units."""
    return predictions * RETURN_SCALE

# Transformer configuration
TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 6

VALUE_MIN = -14
VALUE_MAX = 14

VALUE_SUPPORT_SIZE = VALUE_MAX - VALUE_MIN + 1

VALUE_TRANSFORM_EPSILON = 0.001

class RepresentationNetwork(nn.Module):
    """
    Convert Hidden Hand observations into a learned hidden state
    using a Transformer encoder.

    The observation contains:
        - 8 board cards
        - Discard information
        - Global game information
        - Current decision information
        - 20 previous actions

    The Transformer produces a 512-dimensional hidden state.
    """

    def __init__(self):
        super().__init__()

        # Token embeddings

        # Each card has 4 features
        self.card_embedding = nn.Linear(
            4,
            TRANSFORMER_DIM
        )

        # Discard information has 3 features
        self.discard_embedding = nn.Linear(
            3,
            TRANSFORMER_DIM
        )

        # Global game information has 5 features
        self.global_embedding = nn.Linear(
            5,
            TRANSFORMER_DIM
        )

        # Current decision has 3 features
        self.decision_embedding = nn.Linear(
            3,
            TRANSFORMER_DIM
        )

        # Each action has 6 features and 6 availability masks
        self.action_embedding = nn.Linear(
            ACTION_FEATURES * 2,
            TRANSFORMER_DIM
        )

        # Special CLS token

        # This token collects information from the others
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, TRANSFORMER_DIM)
        )

        # Positional and token-type embeddings

        self.position_embedding = nn.Embedding(
            1 + 8 + 3 + MAX_HISTORY,
            TRANSFORMER_DIM
        )

        self.type_embedding = nn.Embedding(
            5,
            TRANSFORMER_DIM
        )

        # Transformer encoder

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_DIM,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=TRANSFORMER_LAYERS
        )

        # Output projection

        # Convert the Transformer output into the MuZero hidden state
        self.output = nn.Sequential(
            nn.LayerNorm(TRANSFORMER_DIM),
            nn.Linear(TRANSFORMER_DIM, HIDDEN_SIZE)
        )

    def forward(self, observation):

        # Add a batch dimension if necessary
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)

        batch_size = observation.size(0)

        # Verify observation size
        if observation.shape[-1] != OBSERVATION_SIZE:
            raise ValueError(
                f"Expected {OBSERVATION_SIZE} observation features"
            )

        # Separate observation components

        # 8 cards, each containing 4 features
        board = observation[:, 0:32].reshape(
            batch_size,
            8,
            4
        )

        # Discard information
        discard = observation[:, 32:35].unsqueeze(1)

        # Global game information
        global_state = observation[:, 35:40].unsqueeze(1)

        # Current decision information
        decision = observation[:, 40:43].unsqueeze(1)

        # Previous actions
        actions = observation[:, 43:163].reshape(
            batch_size,
            MAX_HISTORY,
            ACTION_FEATURES
        )

        # History availability mask
        history_mask = observation[:, 163:183]

        # Individual action-field masks
        field_masks = observation[:, 183:303].reshape(
            batch_size,
            MAX_HISTORY,
            ACTION_FEATURES
        )

        # Create tokens

        # Combine action values and their availability masks
        action_features = torch.cat(
            [actions, field_masks],
            dim=-1
        )

        # Convert all features to Transformer embeddings
        card_tokens = self.card_embedding(board)

        discard_token = self.discard_embedding(discard)

        global_token = self.global_embedding(global_state)

        decision_token = self.decision_embedding(decision)

        action_tokens = self.action_embedding(action_features)

        # Create one CLS token for each observation
        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        # Combine all tokens
        tokens = torch.cat(
            [
                cls_tokens,
                card_tokens,
                discard_token,
                global_token,
                decision_token,
                action_tokens
            ],
            dim=1
        )

        # Add positional information

        positions = torch.arange(
            tokens.size(1),
            device=observation.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # Token types:
        # 0 = CLS
        # 1 = Board
        # 2 = Discard
        # 3 = Global / Decision
        # 4 = Action history

        token_types = torch.tensor(
            [0] +
            [1] * 8 +
            [2, 3, 3] +
            [4] * MAX_HISTORY,
            device=observation.device
        )

        type_embeddings = self.type_embedding(
            token_types
        )

        tokens = (
            tokens
            + position_embeddings
            + type_embeddings
        )

        # Mask padded history entries

        # CLS + board + discard + global + decision
        # are always available (12 tokens).
        context_mask = torch.zeros(
            batch_size,
            12,
            dtype=torch.bool,
            device=observation.device
        )

        # True means the history position is padding
        history_padding_mask = history_mask == 0

        padding_mask = torch.cat(
            [
                context_mask,
                history_padding_mask
            ],
            dim=1
        )

        # Run Transformer

        contextual_tokens = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask
        )

        # Extract the CLS token's representation
        cls_output = contextual_tokens[:, 0]

        # Produce the final MuZero hidden state
        hidden_state = self.output(cls_output)

        hidden_state = normalize_hidden_state(hidden_state)

        return hidden_state


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

        next_hidden_state = normalize_hidden_state(
            next_hidden_state
        )

        return next_hidden_state, predicted_rewards


class PredictionNetwork(nn.Module):
    """Predict action preferences and categorical future returns."""

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

        # Policy prediction: 7 possible actions
        self.policy_head = nn.Linear(
            HIDDEN_SIZE,
            ACTION_SPACE_SIZE
        )

        # Value prediction: probability distribution for both players
        self.value_head = nn.Linear(
            HIDDEN_SIZE,
            NUM_PLAYERS * VALUE_SUPPORT_SIZE
        )

        # Possible return values: -200 through 200
        self.register_buffer(
            "value_support",
            torch.arange(
                VALUE_MIN,
                VALUE_MAX + 1,
                dtype=torch.float32
            ),
            persistent=False
        )

    def forward_with_logits(self, hidden_state):

        features = self.layers(hidden_state)

        # Predict action preferences
        policy_logits = self.policy_head(features)

        # Predict categorical value distributions
        value_logits = self.value_head(features)

        value_logits = value_logits.reshape(-1, NUM_PLAYERS, VALUE_SUPPORT_SIZE)

        # Convert logits into probabilities
        value_probabilities = torch.softmax(value_logits, dim=-1)

        # Convert distributions into expected numerical values
        # Calculate the expected compressed value
        compressed_values = (
                value_probabilities * self.value_support
        ).sum(dim=-1)

        # Convert compressed values back to original scale
        predicted_values = inverse_transform_value(
            compressed_values
        )

        return policy_logits, predicted_values, value_logits

    def forward(self, hidden_state):

        policy_logits, predicted_values, _ = (self.forward_with_logits(hidden_state))

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

    def initial_inference_with_logits(self, observation):
        """
        Evaluate an observation and return categorical
        value logits for training.
        """

        hidden_state = self.representation(observation)

        policy_logits, values, value_logits = (
            self.prediction.forward_with_logits(hidden_state)
        )

        return hidden_state, policy_logits, values, value_logits

    def recurrent_inference_with_logits(self, hidden_state, action):
        """
        Predict the next hidden state, reward,
        policy, and categorical value logits.
        """

        next_hidden_state, rewards = self.dynamics(
            hidden_state,
            action
        )

        policy_logits, values, value_logits = (
            self.prediction.forward_with_logits(next_hidden_state)
        )

        return (
            next_hidden_state,
            rewards,
            policy_logits,
            values,
            value_logits
        )

def normalize_hidden_state(hidden_state):
    """
    Normalize each hidden-state vector to [0, 1].

    Helps maintain a consistent numerical scale
    during multi-step Dynamics predictions.
    """

    minimum = hidden_state.min(
        dim=-1,
        keepdim=True
    ).values

    maximum = hidden_state.max(
        dim=-1,
        keepdim=True
    ).values

    normalized_state = (
        (hidden_state - minimum)
        / (maximum - minimum).clamp_min(1e-5)
    )

    return normalized_state


def transform_value(value):
    """
    Compress original-scale returns into smaller values.

    Uses MuZero's nonlinear scalar transformation.
    """

    epsilon = VALUE_TRANSFORM_EPSILON

    transformed_value = (
        torch.sign(value)
        * (torch.sqrt(torch.abs(value) + 1.0) - 1.0)
        + epsilon * value
    )

    return transformed_value


def inverse_transform_value(value):
    """
    Convert compressed values back into
    their original numerical scale.

    Used when Prediction outputs values for MCTS.
    """

    epsilon = VALUE_TRANSFORM_EPSILON

    absolute_value = torch.abs(value)

    numerator = (
        torch.sqrt(
            1.0
            + 4.0 * epsilon * (
                absolute_value + 1.0 + epsilon
            )
        )
        - 1.0
    )

    denominator = 2.0 * epsilon

    original_value = (
        torch.sign(value)
        * (
            (numerator / denominator) ** 2
            - 1.0
        )
    )

    return original_value

def normalize_hidden_state(hidden_state):
    """
    Normalize each hidden-state vector to [0, 1].

    Helps maintain a consistent numerical scale
    during multi-step Dynamics predictions.
    """

    minimum = hidden_state.min(
        dim=-1,
        keepdim=True
    ).values

    maximum = hidden_state.max(
        dim=-1,
        keepdim=True
    ).values

    normalized_state = (
        (hidden_state - minimum)
        / (maximum - minimum).clamp_min(1e-5)
    )

    return normalized_state


def transform_value(value):
    """
    Compress original-scale returns into smaller values.

    Uses MuZero's nonlinear scalar transformation.
    """

    epsilon = VALUE_TRANSFORM_EPSILON

    transformed_value = (
        torch.sign(value)
        * (torch.sqrt(torch.abs(value) + 1.0) - 1.0)
        + epsilon * value
    )

    return transformed_value


def inverse_transform_value(value):
    """
    Convert compressed values back into
    their original numerical scale.

    Used when Prediction outputs values for MCTS.
    """

    epsilon = VALUE_TRANSFORM_EPSILON

    absolute_value = torch.abs(value)

    numerator = (
        torch.sqrt(
            1.0
            + 4.0 * epsilon * (
                absolute_value + 1.0 + epsilon
            )
        )
        - 1.0
    )

    denominator = 2.0 * epsilon

    original_value = (
        torch.sign(value)
        * (
            (numerator / denominator) ** 2
            - 1.0
        )
    )

    return original_value
