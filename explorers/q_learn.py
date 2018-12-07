class QLearn(object):
    def __init__(self, q_world, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay_steps=100, discount=0.9, learn_rate=0.1, verbose=True):
        import numpy as np
        from util import LinearDecay

        self.world = q_world
        self.epsilon = LinearDecay(
            epsilon_start, epsilon_end, epsilon_decay_steps)

        self.q_vals = {}
        self.total = {}
        for action in q_world.actions:
            self.q_vals[action] = {}
            for reward in q_world.reward_types:
                self.q_vals[action][reward] = np.zeros(q_world.shape)

            self.total[action] = np.zeros(q_world.shape)

        self.state = None
        self.discount = discount
        self.learn_rate = learn_rate
        self.policy = np.full(self.world.shape, None)
        self.max_qs = np.zeros(q_world.shape)
        self.find_policy()

        self.ep_rewards = {}
        for reward in q_world.reward_types:
            self.ep_rewards[reward] = 0.0
        self.ep_rewards["total"] = 0.0

        self.eps = 1
        self.ep_steps = 0
        self.verbose = verbose

    def __check_ep(self):
        if self.state == None:
            self.state = self.world.reset()

    def find_policy(self):
        import itertools
        import numpy as np

        for (r, c) in itertools.product(range(self.world.shape[0]), range(self.world.shape[1])):
            best_action = None
            max_q = -np.inf

            for action in self.world.actions:
                val = self.total[action][r, c]
                if val > max_q:
                    best_action = action
                    max_q = val

            assert(best_action != None)
            self.policy[r, c] = best_action
            self.max_qs[r, c] = max_q

    def observe(self, action, rewards, next_state):
        a = self.learn_rate
        y = self.discount

        for r_type, obs in rewards.items():
            self.q_vals[action][r_type][self.state] = (
                1 - a) * self.q_vals[action][r_type][self.state] + \
                a * y * \
                (obs + self.q_vals[self.policy[next_state]]
                 [r_type][next_state])

    def update_qs(self, action):
        total = 0.0
        for r_map in self.q_vals[action].values():
            total += r_map[self.state]

        self.total[action][self.state] = total

    def act(self):
        import random

        self.__check_ep()
        self.ep_steps += 1

        roll = random.random()
        action = None
        if roll < self.epsilon():
            action = random.choice(self.world.actions)
        else:
            action = self.policy[self.state]

        next_state, rewards, total, terminal = self.world.act(
            self.state, action)

        self.ep_rewards["total"] += total

        self.observe(action, rewards, next_state)

        self.update_qs(action)

        if terminal:
            self.state = None
            self.end_ep()
        else:
            self.state = next_state

        self.find_policy()

    def run_ep(self):
        self.state = self.world.reset()
        self.steps = 0
        self.__clean_rewards()

        while self.state != None:
            self.act()

    def run_until(self, stop_condition):
        while not stop_condition(self):
            self.run_ep()

    def run_fixed_eps(self, num_eps=150):
        eps = 0

        def ep_counter(_):
            nonlocal eps
            eps += 1
            return eps >= num_eps

        self.run_until(ep_counter)

    def __clean_rewards(self):
        for reward in self.world.reward_types:
            self.ep_rewards[reward] = 0.0
        self.ep_rewards["total"] = 0.0

    def end_ep(self):
        if self.verbose:
            print("Ended episode %d with reward %d in %d steps" %
                  (self.eps, self.ep_rewards["total"], self.ep_steps))

        self.eps += 1
        self.steps = 0
        self.__clean_rewards()

    def format_solution(self):
        from prettytable import PrettyTable, ALL
        fmt_str = "Ran Q-learning for %d episodes w/ gamma = %f, alpha = %f" % (
            self.eps, self.discount, self.learn_rate)

        table = PrettyTable([])
        table.hrules = ALL
        table.add_column("Reward Type", [*self.world.reward_types, "total"])
        for action in self.world.actions:
            table.add_column(
                action, [*self.q_vals[action].values(), self.total[action]])

        fmt_str = "%s\n\nValues:\n%s\n\nMaxQ:\n%s\n\nPolicy:\n%s" % (
            fmt_str, table, self.max_qs, self.policy)

        return fmt_str


class QPolicy(object):
    def __init__(self, q_learn):
        self.__learner = q_learn
