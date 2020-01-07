import numpy as np
from Game import *


class StateAggregator:
    def __init__(self, aggregate_factor=9):
        self.aggregate_factor = aggregate_factor

    def aggregate(self, game: Game):
        aggregated_states = np.arange(game.get_n_states() / self.aggregate_factor)
        mapping = []
        for i in range(game.get_n_states()):
            mapping.append([[], [], []])  # 0 for periphery, 1 for boundary, 2 for interior
        for original_state in range(game.get_n_states()):
            aggregated_index = int(original_state / self.aggregate_factor)
            is_periphery = False
            is_boundary = False
