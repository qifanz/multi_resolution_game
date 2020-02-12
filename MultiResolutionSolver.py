from NashSolver import NashSolver
from StateAggregator import StateAggregator


class MultiResolutionSolver:
    def __init__(self, game, aggregate_factor):
        self.game = game
        self.aggregate_factor = aggregate_factor

    def solve(self):
        state_aggregator = StateAggregator(self.game, self.aggregate_factor)
        aggregated_game = state_aggregator.aggregate()
        nash_solver = NashSolver(aggregated_game)
        return nash_solver.solve()
