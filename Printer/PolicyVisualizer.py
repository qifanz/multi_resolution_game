import pickle
import sys
from Game import Game
import Actions

POLICY_FILE = '../data/policy_5_5.pkl'


def import_policy():
    f = open(POLICY_FILE, 'rb')
    policy = pickle.load(f)
    f.close()
    return policy


if __name__ == '__main__':
    state = int(sys.argv[1])

    game = Game()
    game.print_state(state)

    policy = (import_policy())[state]
    policy_x = policy[0]
    policy_y = policy[1]

    print('------ Player 1 ------')
    for i in range(len(policy_x)):
        print(Actions.get_str(i), ' : ', str(round(policy_x[i], 5)))
    print('------ Player 2 ------')
    for i in range(len(policy_y)):
        print(Actions.get_str(i), ' : ', str(round(policy_y[i], 5)))
