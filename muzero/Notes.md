# Muzero Model Architecture Notes

## Terms
* Latent State - a vector of numbers used to represent a particular state (Model determines the meaning during training similar to an embedding)

## Base Model
The base muzero model has 3 different parts:
1. Representation
2. Dynamics
3. Prediction

### Representation Model
The representation model is a transformer model that works to turn the state inputs from the environment into inputs for the model, which are tokenized observation features from the environment.  It interprets the current envrionment state (which is thes observation features) into a language the other models are able to more easily interpret.  Can be thought of as embedding the state for easier representation in the future models.  The transformer model here encodes the inputs and then outputs a single vector representing that current state. 

### Dynamics Model
The Dynamics Model takes in the single vector (latent state), and a one-hot encoded action along with it.  From this it outputs a latent state and a reward given the input latent state.  The reward output is a 2-vector output with a number reresenting the player who acted and other number representing the other player in the game. 

The model itself is a single step transition model which means it calculates a single move at a time.  This is used by the Monte Carlo Tree Search to move forward in the latent space so that we do not need to utilize the current environment while searching for a optimal reward.

The Dynamics model does not look at all possible actions, but only the actions that are passed to it.  The actions passed to the Dynamics model is determined by the MCTS.  It does this by only passing legal moves to the model at the root of the Tree. Legal-move masking applies only at the root; deeper in the tree all actions are allowed.  

### Predictions Model
The prediction model looks at the current latent state and ouputs policy logits and a 2-vector value (expected future return, not a reward).  This value is a 2 vector representation of our current state as the controlled player, and our opponents current state to represent the expected return of the 2 players.  However, the difference between this model and the dynamics model is that this model is looking at the total returns from the current latent state onward into the future.  TLDR - allows the system to look at the long term rather than just the next step in the Dynamics Model to allow the system to plan a game out.  

These outputs help in the MCTS to determine which branch to go down (action to take next).  As the model learns a better policy, the more the model will reward good decisions (or decisions that get us closer to winning the game).  

Prediction also outputs a policy, which Dynamics has no equivalent of. This is the prior (the network's pre-search guess at good moves) used to steer the search in the tree. 

## Surrounding the Models

### Monte Carlo Tree Search (MCTS)
This is the planning part of the Muzero archtiecture.  This tree allows us to look into the future at next decisions, while continuing to keep track of the decisions that the muzero model has already made in order to shape the policy in the Predictions layer

1. Start at the current position (the root of the tree).
2. Pick a promising move, using the network's policy prior and what past simulations found.
3. Use Dynamics to imagine the position after that move.
4. Use Prediction to judge that imagined position.
5. Pass that judgment back up the path, so every move along it learns "this line looked good (or bad)."
6. Repeat, 16 times in the config (num_simulations, mcts.py:27).

A fresh tree is built for each move, and the real move is picked from the root visit counts.

## Training
Traditional Muzero just plays against itself (self-play); for SkyNet training specifically we used curriculum learning with bots. It plays with the current network and the MCTS, and as it plays it addds the games to a replay buffer to view the previous game/step outcomes. The buffer stores the MCTS visit counts, since the policy target is the search's visit counts, not the move played.  From here is then takes steps in gradient decent to minimize prediction error (it does not directly maximize reward): policy is trained toward the MCTS visit counts, value toward the actual returns, and reward toward the observed rewards. 

During training we unroll the Dynamics model for 5 steps. Starting from the latent state (step 0) produced by Representation, we apply Dynamics using the actions that were actually played during a game, feeding each output state back in as the next input. At every step, the predicted policy, value and reward are compared against the real game's results. This trains Dynamics to stay accurate over multiple steps of its own predictions, which is how it's used in MCTS. Because the loss only checks policy, value and reward, and never compares to the real game state, the latent states end up holding whatever information those predictions need. 

## Encoding
The observation becomes 31 tokens: 8 board cards, a discard token, a global token, a decision token, and up to 20 history tokens. Everything is expressed relative to the observer ("own" vs "opponent"), so the same weights work for both seats. It also uses obs["tokens"] only, which contains nothing hidden.

## Limitations
Card games are a known weak spot for Muzero: hidden cards and random draws break its deterministic assumptions, so this is probably not a good example to reference for Muzero.

## Temperature
Controls the exploration the model does during training.  Makes the model more random at first, and then more greedy as training moves on. 