class AggStates:
    def __init__(self, n_aggregated_states, n_absorbing_type):
        '''
        constructor
        :param n_aggregated_states: number of aggregated states EXCLUDE absorbing states
        '''
        self.n_aggregated_states = n_aggregated_states
        self.mapping = []
        self.states = []
        self.n_absorbing_type = n_absorbing_type
        self.absorbing_state = []
        for i in range(n_aggregated_states):
            self.mapping.append([set(), set(), set()])
            self.states.append(set())
        for i in range(n_absorbing_type):
            self.absorbing_state.append([])

    def get_total_state_count(self):
        '''
        Get total number of aggregated states, INCLUDE absorbing states
        :return: number of aggregated states
        '''
        return self.n_absorbing_type + self.n_aggregated_states

    def add_absorbing_state(self, absorbing_type, state):
        self.absorbing_state[absorbing_type].append(state)

    def add_boundary_state(self, aggregated_index, state):
        self.mapping[aggregated_index][1].add(state)
        self.states[aggregated_index].add(state)

    def add_interior_state(self, aggregated_index, state):
        self.mapping[aggregated_index][2].add(state)
        self.states[aggregated_index].add(state)

    def add_periphery_state(self, aggregated_index, state):
        self.mapping[aggregated_index][0].add(state)

    def get_boundary_state(self, aggregated_index):
        return self.mapping[aggregated_index][1]

    def get_interior_state(self, aggregated_index):
        return self.mapping[aggregated_index][2]

    def get_periphery_state(self, aggregated_index):
        return self.mapping[aggregated_index][0]

    def get_all_states_in_index(self, aggregated_index):
        return self.states[aggregated_index]

    def is_terminal_state(self, aggregated_index):
        return aggregated_index >= self.n_aggregated_states