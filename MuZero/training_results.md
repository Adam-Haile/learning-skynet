
# Before adding other MCTS actions and with 1000 replay buffer

| Iteration | Win Rate/Average Score|
|-|-|
| 10 | 39.4/31.551|
| 20 | 36.55/32.8055|
| 30 | 45.95/39.6645|
| 40 | 43.25/30.453|
| 50 | 50.55/28.53 (Best)|
| 100 | 35.75/32.762|

Note from the graph we had a sharp decline at 50 iterations (1000 games) so we should test a larger buffer

# Before adding other MCTS actions and with 2000 replay buffer

| Iteration | Win Rate/Average Score|
|-|-|
| 10 | 39.5/31.546|
| 20 | 35.4/32.954|
| 30 | 42.8/31.3705|
| 40 | 42.3/30.756|
| 50 | 42/31.1135|
| 100 | 44.5/29/7615 (Best)|

this model did not reach the same performance as the previous one but the sharp decline at 1000 games disappeared. This model continued to get better over the entire course of the training.

We should analyze different seeds to confirm this pattern.

# Updated MCTS tree to include other actions

1k buffer

[{'wins': 706, 'losses': 1294, 'ties': 0, 'win_rate': 0.353, 'average_score': 33.2635, 'iteration': 10}, {'wins': 644, 'losses': 1356, 'ties': 0, 'win_rate': 0.322, 'average_score': 35.389, 'iteration': 20}, {'wins': 734, 'losses': 1266, 'ties': 0, 'win_rate': 0.367, 'average_score': 32.701, 'iteration': 30}, {'wins': 755, 'losses': 1245, 'ties': 0, 'win_rate': 0.3775, 'average_score': 32.421, 'iteration': 40}, {'wins': 769, 'losses': 1231, 'ties': 0, 'win_rate': 0.3845, 'average_score': 32.0555, 'iteration': 50}, {'wins': 718, 'losses': 1282, 'ties': 0, 'win_rate': 0.359, 'average_score': 32.7035, 'iteration': 100}]

2k buffer

[{'wins': 797, 'losses': 1203, 'ties': 0, 'win_rate': 0.3985, 'average_score': 31.598, 'iteration': 10}, {'wins': 760, 'losses': 1240, 'ties': 0, 'win_rate': 0.38, 'average_score': 32.6095, 'iteration': 20}, {'wins': 688, 'losses': 1312, 'ties': 0, 'win_rate': 0.344, 'average_score': 33.4605, 'iteration': 30}, {'wins': 945, 'losses': 1055, 'ties': 0, 'win_rate': 0.4725, 'average_score': 29.5915, 'iteration': 40}, {'wins': 807, 'losses': 1193, 'ties': 0, 'win_rate': 0.4035, 'average_score': 32.8885, 'iteration': 50}, {'wins': 870, 'losses': 1130, 'ties': 0, 'win_rate': 0.435, 'average_score': 30.7315, 'iteration': 100}]

# Trying to meet the requirements

after running the model against the three bots these were the winrates


| Opponent | Win Rate (%) | Minimum (%) | Average Score | Status |
|-|-|-|-|-|
|random|36.40	| 36.0|32.73|PASS|
|greedy|40.65	| 45.0|31.10|FAIL|
|tell_reader|22.00 | 20.0 | 32.55 | PASS|

we're failing against greedy.

Chosen action: TRADE(2)

KEEP | Visits: 9 | Prior: 0.096 | Estimated value: -22.971
PEEK | Visits: 9 | Prior: 0.102 | Estimated value: -22.972
TRADE(0) | Visits: 5 | Prior: 0.111 | Estimated value: -22.929
TRADE(1) | Visits: 9 | Prior: 0.153 | Estimated value: -23.016
TRADE(2) | Visits: 17 | Prior: 0.256 | Estimated value: -22.89
TRADE(3) | Visits: 15 | Prior: 0.283 | Estimated value: -22.929

All the predicted values are very similar. This could hurt as it only thinks that trade(2) is very slightly better than keep

Our experiments show that the MuZero agent is learning to play Hidden Hand, but its performance remains inconsistent across training configurations. Increasing the replay buffer from 1,000 to 2,000 games generally produced better results, while expanding MCTS to consider KEEP, PEEK, TRADE, and CALL did not consistently improve win rates. Increasing MCTS simulations also showed mixed results, suggesting that additional search alone is insufficient. The agent currently meets the minimum benchmarks against random and tell_reader opponents but remains approximately 4 percentage points below the 45% target against greedy. Analysis of the neural networks revealed that predicted action values are very similar across different moves, while terminal reward predictions can have substantial errors. These findings suggest that improving reward and value prediction accuracy, particularly for terminal states, should be our next priority, although further testing is needed to confirm the causes of the performance limitations.

After changing from MSE to Huber loss function, decreasing learning rate, and increasing the # of weight adjustments from 20 to 40 per game this made us meet the targets.