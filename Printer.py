from math import floor
import numpy as np
import sys
from Game import Game

if __name__ == '__main__':
    state = int(sys.argv[1])
    game = Game()
    game.print_state(state)
