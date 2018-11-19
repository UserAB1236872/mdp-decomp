import numpy as np
import itertools


class Gridworld(object):
    def __init__(self, rewards, terminals, misfires, impassable, world_shape, name, misfire_prob=0.1):
        self.__terminals = terminals
        assert(terminals.shape == world_shape)
        self.__misfires = misfires
        assert(misfires.shape == world_shape)
        self.__impassable = impassable
        assert(impassable.shape == world_shape)
        self.__rewards = rewards

        self.__shape = world_shape
        self.__total_reward = np.zeros(world_shape)
        for _, vals in self.rewards.items():
            self.__total_reward += vals
            assert(world_shape == vals.shape)

        self.__actions = ['u', 'd', 'l', 'r']
        self.__misfire_prob = misfire_prob
        self.name = name

    @property
    def terminals(self):
        return self.__terminals

    @property
    def misfires(self):
        return self.__misfires

    @property
    def impassable(self):
        return self.__impassable

    @property
    def rewards(self):
        return self.__rewards

    @property
    def shape(self):
        return self.__shape

    @property
    def total_reward(self):
        return self.__total_reward

    @property
    def actions(self):
        return self.__actions

    @property
    def misfire_prob(self):
        return self.__misfire_prob

    @property
    def states(self):
        return filter(lambda x: not self.impassable[x], itertools.product(range(self.shape[0]), range(self.shape[1])))

    def nonterminal_states(self):
        return filter(lambda x: not self.terminals[x], self.states)

    def successors(self, state, action):
        return self.succ_map(state)[action]

    def succ_map(self, state):
        succs = {key: [val]
                 for key, val in self.__direct_succ_map(state).items()}

        if self.misfires[state] == None or len(self.misfires[state]) == 0:
            return succs

        for misfire in self.misfires[state]:
            for succ in succs.keys():
                if succ == misfire:
                    continue

                if succs[misfire] not in succs[succ]:
                    succs[succ].append(succs[misfire][0])

        return succs

    def all_successors(self, state):
        return set(self.__direct_succ_map(state).values())

    def __direct_succ_map(self, state):
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
        succs = self.__direct_succ_map(state)
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
        return np.fromfunction(
            np.vectorize(lambda x, y: self.transition_prob(state, action, (int(x), int(y)))), self.shape)

    def __printable_elem(self, val, x, y):
        coord = (x, y)
        if self.impassable[coord]:
            return 'x'
        elif self.misfires[coord] != None and len(self.misfires[coord]) > 0:
            return self.misfires[coord]
        elif val[coord] == 0:
            return ' '
        else:
            return str(val[coord])

    def printable(self):
        out = {}

        for typ, val in self.rewards.items():
            out[typ] = np.fromfunction(
                np.vectorize(lambda x, y: self.__printable_elem(val, int(x), int(y))), self.shape)

        out["total"] = np.fromfunction(
            np.vectorize(lambda x, y: self.__printable_elem(self.total_reward, int(x), int(y))), self.shape)

        return out
