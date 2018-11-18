class Gridworld(object):
    def __init__(self, rewards, terminals, misfires, impassable, world_shape, misfire_prob=0.1):
        import numpy as np

        self.terminals = terminals
        assert(terminals.shape == world_shape)
        self.misfires = misfires
        assert(misfires.shape == world_shape)
        self.impassable = impassable
        assert(impassable.shape == world_shape)
        self.rewards = rewards

        self.shape = world_shape
        self.total_reward = np.zeros(world_shape)
        for _, vals in self.rewards:
            self.total_reward += vals
            assert(world_shape == vals.shape)

        self.actions = ['u', 'd', 'l', 'r']
        self.misfire_prob = misfire_prob

    def states(self):
        import itertools
        return itertools.product(range(self.shape[0]), range(self.shape[1]))

    def successors(self, state):
        return set([*self.succ_map(state).values()])

    def succ_map(self, state):
        succs = {}
        if state[0] - 1 >= 0 and not self.impassable[state[0] - 1, state[1]]:
            succs['u'] = (state[0] - 1, state[1])
        else:
            succs['u'] = state

        if state[0] + 1 < self.shape[0] and not self.impassable[state[0] + 1, state[1]]:
            succs['d'] = (state[0] + 1, state[1])
        else:
            succs['d'] = state

        if state[1] - 1 >= 0 and not self.impassable[state[0], state[1] - 1]:
            succs['l'] = (state[0], state[1] - 1)
        else:
            succs['l'] = state

        if state[1] + 1 < self.shape[1] and not self.impassable[state[0], state[1] + 1]:
            succs['r'] = (state[0], state[1] + 1)
        else:
            succs['r'] = state

        return succs

    def transition_prob(self, state, action, nxt):
        succs = self.succ_map(state)
        assert(action in self.actions)
        if nxt not in succs.values() or self.impassable[nxt]:
            return 0.0

        misfires = self.misfires[state]
        if misfires == None:
            misfires = ''
        misfires = misfires.replace(action, '')

        if nxt == succs[action]:
            return 1.0 - len(misfires) * self.misfire_prob

        for a, succ in succs.items():
            if succ == nxt and a in misfires:
                return self.misfire_prob

        return 0.0

    def transition_matrix(self, state, action):
        import numpy as np

        return np.fromfunction(
            lambda succ: self.transition_prob(state, action, succ), self.shape)
