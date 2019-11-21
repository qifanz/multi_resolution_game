class Game:
    def __init__(self):
        self.states, self.state_index_mapping = self.initialize_states()
        self.rewards = self.initialize_rewards()
        self.transitions = self.initialize_transitions()

    def get_n_states(self):
        return len(self.states)

    def get_state_reward(self, state):
        return self.rewards[state]

    def get_state_transition(self, action_pair):
        return self.transitions[action_pair[0]][action_pair[1]]

    def get_state(self, state):
        return self.state_index_mapping[state]

    def initialize_states(self):
        return [],[]

    def initialize_rewards(self):
        return []

    def initialize_transitions(self):
        return []