N_ACTIONS = 4


def get_movement(action):
    assert (action < N_ACTIONS)
    if action == 0:
        return (-1, 0)
    elif action == 1:
        return (0, -1)
    elif action == 2:
        return (1, 0)
    elif action == 3:
        return (0, 1)
