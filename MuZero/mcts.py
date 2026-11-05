import math
import torch

class Node:
    def __init__(self, prior=0.0):
        # Probability of selecting the action leading to this node
        self.prior = prior

        # Number of times this node has been visited
        self.visit_count = 0

        # Store the total value for each player
        self.value_sum = [0.0, 0.0]

        # Possible actions and their corresponding child nodes
        self.children = {}

        # Learned representation of this state
        self.hidden_state = None

        # Predicted reward for reaching this state
        self.reward = None

        # Player who makes the next decision at this state
        self.player = None

        self.terminal = False

    def value(self, player):
        if self.visit_count == 0:
            return 0.0

        return self.value_sum[player] / self.visit_count

def select_child(node, player, c_puct=1.5, discount=1.0):

    best_score = float("-inf")
    best_action = None
    best_child = None

    # Check every possible action from this node
    for action in node.children:

        child = node.children[action]

        value = discount * child.value(player)

        if child.reward is not None:
            value = child.reward[player] + value

        # Calculate how much we should explore this action
        parent_visits = math.sqrt(node.visit_count + 1)
        child_visits = child.visit_count + 1

        exploration = c_puct * child.prior * parent_visits / child_visits

        # Calculate the total score
        score = value + exploration

        # Keep track of the action with the highest score
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child

def expand_node(node, hidden_state, reward, policy_logits, legal_actions):

    # Store the information for this node
    node.hidden_state = hidden_state
    node.reward = reward

    # Convert action scores into probabilities
    probabilities = torch.softmax(policy_logits, dim=-1)

    # Remove the batch dimension
    probabilities = probabilities.squeeze(0)

    # Only keep probabilities for legal actions
    legal_probabilities = probabilities[legal_actions]

    # Make legal probabilities add up to 1
    legal_probabilities = legal_probabilities / legal_probabilities.sum()

    # Create a child node for each legal action
    for i in range(len(legal_actions)):

        action = legal_actions[i]
        probability = legal_probabilities[i].item()

        node.children[action] = Node(prior=probability)

def backpropagate(search_path, value, discount=1.0):
    """
    Update the visited nodes after an MCTS simulation.

    search_path: Nodes visited during the simulation.
    value: Predicted future rewards for both players.
    discount: How much future rewards matter.
    """
    current_values = [value[0], value[1]]

    # Work backward from the last node to the root
    for node in reversed(search_path):

        # Update the node using its future value
        node.visit_count += 1

        node.value_sum[0] += current_values[0]
        node.value_sum[1] += current_values[1]

        # Include the reward for reaching this node
        # before moving to its parent
        if node.reward is not None:
            reward = node.reward

            current_values[0] = reward[0] + discount * current_values[0]
            current_values[1] = reward[1] + discount * current_values[1]

def run_mcts(model, encoded_observation, legal_actions, current_player,
             num_simulations=16, max_depth=4, discount=1.0):
    """Search from one real observation; return the selected action and tree.

    This baseline assumes the default Hidden Hand rules:
    - TRADE(0..3) remain legal at simulated nonterminal states.
    - Players normally alternate after each action.
    - CALL (action 6) ends the game immediately.
    """
    # Check that the inputs are valid
    if len(legal_actions) == 0:
        raise ValueError("No legal actions available.")

    if current_player not in [0, 1]:
        raise ValueError("Invalid player.")

    if num_simulations < 1 or max_depth < 1:
        raise ValueError("Simulations and depth must be positive.")

    # Add the batch dimension if needed
    if encoded_observation.ndim == 1:
        encoded_observation = encoded_observation.unsqueeze(0)

    # Disable training behavior during MCTS
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():

            # STEP 1: Create the root node
            hidden_state, policy, value = model.initial_inference(
                encoded_observation
            )

            root = Node()
            root.player = current_player

            expand_node(root, hidden_state, None, policy, legal_actions)

            # STEP 2: Run MCTS simulations
            for simulation in range(num_simulations):

                node = root
                search_path = [root]

                # STEP 3: Explore the search tree
                for depth in range(max_depth):

                    # Stop if this state ends the game
                    if node.terminal:
                        leaf_value = [0.0, 0.0]
                        break

                    # Stop if there are no actions to explore
                    if len(node.children) == 0:
                        leaf_value = [0.0, 0.0]
                        break

                    # Select the next action
                    action, child = select_child(node, node.player)

                    parent = node
                    node = child

                    search_path.append(node)

                    # STEP 4: Expand a new node
                    if node.hidden_state is None:

                        action_tensor = torch.tensor([action], device=parent.hidden_state.device)

                        next_state, rewards, policy, values = (
                            model.recurrent_inference(parent.hidden_state, action_tensor)
                        )

                        node.player = 1 - parent.player

                        reward_list = rewards[0].tolist()

                        # CALL ends the game
                        if action == 6:
                            node.terminal = True
                            future_actions = []
                            leaf_value = [0.0, 0.0]

                        else:
                            future_actions = [2, 3, 4, 5]
                            leaf_value = values[0].tolist()

                        expand_node(node, next_state, reward_list, policy, future_actions)

                        break

                else:
                    # Maximum search depth was reached
                    policy, values = model.prediction(node.hidden_state)
                    leaf_value = values[0].tolist()

                # STEP 5: Update the search tree
                backpropagate(search_path, leaf_value, discount)

            # STEP 6: Choose the most-visited action
            best_action = None
            most_visits = -1

            for action in root.children:

                child = root.children[action]

                if child.visit_count > most_visits:
                    most_visits = child.visit_count
                    best_action = action

            return best_action, root

    finally:
        model.train(was_training)