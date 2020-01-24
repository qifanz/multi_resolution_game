from Game import *
from NashSolver import *
from NashValidator import *
from StateAggregator import *
import time

start = time.time()
game = Game()
end = time.time()
print('Init game used ', end - start)
#state_aggregator = StateAggregator()
#state_aggregator.aggregate(game)
start = time.time()
nash_solver = NashSolver(game)
nash_solver.solve()
end = time.time()
print('Nash Solver used ', end - start)
#nash_validator = NashValidator(game)
#nash_validator.solve()
#f = open('value.pkl', 'rb')
#value = pickle.load(f)
#f.close()

#f = open('value_validator.pkl', 'rb')
#value_validator = pickle.load(f)
#f.close()

print('done')

#testing github setting
