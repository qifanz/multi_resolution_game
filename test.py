from Game import *
from NashSolver import *
from NashValidator import *
from StateAggregator import *

game = Game()
state_aggregator = StateAggregator()
state_aggregator.aggregate(game)

nash_solver = NashSolver(game)
# nash_solver.solve()
nash_validator = NashValidator(game)
nash_validator.solve()
f = open('value.pkl', 'rb')
value = pickle.load(f)
f.close()

f = open('value_validator.pkl', 'rb')
value_validator = pickle.load(f)
f.close()

print('done')
