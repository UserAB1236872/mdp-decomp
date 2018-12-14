class LinearDecay(object):
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.curr_steps = 0
        self.curr_epsilon = start

    def __call__(self):
        self.curr_steps += 1
        interp = min(self.curr_steps, self.steps) / self.steps
        self.curr_epsilon = (1 - interp) * self.start + interp * self.end

        return self.curr_epsilon


class Sarsa(object):
    def __init__(self, q_world, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay_steps=100, discount=0.9, learn_rate=0.01, verbose=True, max_episodes=10000):
        import numpy as np
        from util import LinearDecay

        self.world = q_world
        # self.epsilon = LinearDecay(
        #     epsilon_start, epsilon_end, epsilon_decay_steps)

        self.epsilon = epsilon_start
        self.max_eps = epsilon_start
        self.min_eps = epsilon_end
        self.max_episodes = max_episodes
        self.curr_episode = 0
        self.threshold_eps = 0.8 * max_episodes

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
        self.action = None

    def __check_ep(self):
        if self.state == None:
            self.state = self.world.reset()
            self.action = self.__epsilon_greedy(self.state)

    def __epsilon_greedy(self, state):
        import random
        roll = random.random()
        if roll < self.epsilon:
            return random.choice(self.world.actions)
        else:
            return self.policy[state]

    def find_policy(self):
        import itertools
        import numpy as np

        for state in itertools.product(range(self.world.shape[0]), range(self.world.shape[1])):
            best_action = None
            max_q = -np.inf

            for action in self.world.actions:
                val = self.total[action][state]
                if val > max_q:
                    best_action = action
                    max_q = val

            assert(best_action != None)
            self.policy[state] = best_action
            self.max_qs[state] = max_q

    def act(self):
        self.__check_ep()
        self.ep_steps += 1

        a = self.learn_rate
        y = self.discount

        next_state, rewards, total, terminal = self.world.act(
            self.state, self.action)
        next_action = self.__epsilon_greedy(next_state)

        self.ep_rewards["total"] += total

        total = 0.0
        for (reward, obs) in rewards.items():
            self.q_vals[self.action][reward][self.state] = self.q_vals[self.action][reward][self.state] + a * (
                y * obs + y * self.q_vals[next_action][reward][next_state] - self.q_vals[self.action][reward][self.state])
            total += self.q_vals[self.action][reward][self.state]

        self.total[self.action][self.state] = total

        if terminal:
            self.state = None
            self.end_ep()
        else:
            self.state = next_state
            self.action = next_action

        self.find_policy()

    def run_ep(self):
        self.state = self.world.reset()
        self.action = self.__epsilon_greedy(self.state)
        self.steps = 0
        self.__clean_rewards()

        while self.state != None:
            self.act()

    def run_until(self, stop_condition):
        while not stop_condition(self):
            self.curr_episode += 1
            self.run_ep()
            self.epsilon = max(self.min_eps,
                               self.max_eps * ((self.threshold_eps - self.curr_episode)/self.threshold_eps))

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
        fmt_str = "Ran SARSA for %d episodes w/ gamma = %f, alpha = %f" % (
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
