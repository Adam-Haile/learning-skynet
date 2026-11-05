import random

class ReplayBuffer:

    def __init__(self, max_games=1000):

        # Maximum number of games we can store
        self.max_games = max_games

        # List containing completed games
        self.games = []

    def add_game(self, game_history):

        # Do not store empty games
        if len(game_history) == 0:
            raise ValueError("Cannot store an empty game")

        # Store the completed game
        self.games.append(game_history)

        # Remove the oldest game if the buffer is full
        if len(self.games) > self.max_games:
            self.games.pop(0)

    def sample_games(self, batch_size):

        # Check whether we have enough games
        if batch_size > len(self.games):
            raise ValueError("Not enough games in the replay buffer")

        # Randomly select games
        return random.sample(self.games, batch_size)

    def __len__(self):
        return len(self.games)