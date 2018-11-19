import numpy as np
import itertools

from prettytable import PrettyTable, ALL
from numpy.linalg import norm


class ValueIter(object):
    def __init__(self, world, beta=0.9):
        self.beta = beta
        self.prev_total = np.full(world.shape, np.inf)
        self.world = world
        self.values = {}
        self.total = world.total_reward.copy()
        for (k, v) in world.rewards.items():
            self.values[k] = v.copy()

        self.policy = np.full(world.shape, None)
        self.iters = 0
        self.find_policy()

    def find_policy(self):
        for state in self.world.states:
            if self.world.terminals[state]:
                self.policy[state] = None
                continue

            policy = None
            max_action_val = -np.inf

            for action in self.world.actions:
                total = 0.0
                for nxt in self.world.successors(state, action):
                    total += self.total[nxt] * \
                        self.world.transition_prob(
                            state, action, nxt)

                if total > max_action_val:
                    max_action_val = total
                    policy = action

            self.policy[state] = policy

    def update(self):
        self.iters += 1
        rows = self.world.shape[0]
        cols = self.world.shape[1]

        new_vals = {}
        for (typ, vals) in self.values.items():
            typ_vals = np.zeros(self.world.shape)
            for state in self.world.states:
                for nxt in self.world.successors(self.policy[state]):
                    typ_vals[state] += self.world.transition_prob(
                        state, self.policy[state], nxt) * vals[nxt]

                typ_vals[state] *= self.beta
                typ_vals[state] += self.world.rewards[typ][state]

            new_vals[typ] = typ_vals

        self.values = new_vals

        (self.prev_total, self.total) = (self.total, self.prev_total)
        self.total = np.zeros(self.world.shape)
        for (_, v) in self.values.items():
            self.total += v

        # Update policy
        self.find_policy()

    def update_until(self, stop_condition):
        """
        condition is a function
        """
        while not stop_condition(self):
            self.update()

        return self.values

    def solve(self, max_iters=None, threshold=1e-8):
        # We want to use a local iters variable
        # or you get counterintuitive behavior
        iters = 0

        def tolerance(slf):
            nonlocal iters
            if max_iters != None and iters > max_iters:
                return True
            iters += 1
            return norm(slf.total - slf.prev_total, ord=np.inf) < threshold
        return self.update_until(tolerance)

    def format_solution(self):
        fmt_str = "Solved problem in %d iterations w/ beta=%f" % (
            self.iters, self.beta)

        table = PrettyTable([])
        table.hrules = ALL
        table.add_column("Reward Type", [*self.world.rewards, "total"])
        table.add_column(
            "Values", [*self.values.values(), self.total])

        fmt_str = "%s\n\nValues:\n%s\n\nPolicy:\n%s" % (
            fmt_str, table, self.policy)

        return fmt_str
