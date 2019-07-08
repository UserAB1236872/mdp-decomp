import os
import matplotlib as mpl

mpl.use('Agg')  # to plot graphs over a server.
import matplotlib.pyplot as plt


class QLearnPresFuture(object):
    def __init__(self, q_world, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay_steps=4000, discount=0.9,
                 learn_rate=0.01, verbose=True, max_episodes=10000):
        import numpy as np
        from util import LinearDecay

        self.world = q_world
        # self.epsilon = LinearDecay(
        #     epsilon_start, epsilon_end, epsilon_decay_steps)

        self.epsilon = epsilon_start
        self.max_eps = epsilon_start
        self.min_eps = epsilon_end
        self.max_episodes = max_episodes

        self.threshold_eps = 0.8 * self.max_episodes

        self.present = {}
        self.future = {}
        self.total = {}

        for a in self.world.actions:
            self.present[a] = np.zeros(self.world.shape)
            self.future[a] = np.zeros(self.world.shape)
            self.total[a] = np.zeros(self.world.shape)

        self.state = None
        self.discount = discount
        self.learn_rate = learn_rate
        self.policy = np.full(self.world.shape, None)
        self.max_qs = np.zeros(q_world.shape)
        self.find_policy()

        self.ep_rewards = 0.0

        self.eps = 1
        self.ep_steps = 0
        self.verbose = verbose

        self.epses = []

        self.curr_episode = 0

    def __check_ep(self):
        if self.state is None:
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

            assert (best_action is not None)
            self.policy[r, c] = best_action
            self.max_qs[r, c] = max_q

    def observe(self, action, rewards, next_state, terminal):
        a = self.learn_rate
        y = self.discount

        reward = sum(rewards.values())
        self.present[action][self.state] = (1-a) * self.present[action][self.state] + a * reward

        if not terminal:
            next_action = self.policy[next_state]
            future_target = self.present[next_action][next_state] + self.future[next_action][next_state]
            self.future[action][self.state] = (1-a) * self.future[action][self.state] + a * y * future_target
        else:
            self.future[action][self.state] = (1-a) * self.future[action][self.state] + 0

    def update_qs(self, action):
        self.total[action][self.state] = self.present[action][self.state] + self.future[action][self.state]

    def act(self):
        import random

        self.__check_ep()
        self.ep_steps += 1

        roll = random.random()
        eps = self.epsilon
        if roll < eps:
            action = random.choice(self.world.actions)
        else:
            action = self.policy[self.state]
        next_state, rewards, total, terminal = self.world.act(self.state, action)

        self.ep_rewards += total

        self.observe(action, rewards, next_state, terminal)

        self.update_qs(action)

        if terminal:
            self.state = None
            self.end_ep()
        else:
            self.state = next_state

        self.epses.append(eps)

        self.find_policy()

    def run_ep(self):
        self.state = self.world.reset()
        self.steps = 0
        self.__clean_rewards()
        # ep_length = 0
        while self.state is not None:
            self.act()
            # ep_length += 1
        # print('ep_length: ', ep_length)
        return self.steps

    def run_until(self, stop_condition):
        total_steps = 0
        start_ep = self.epsilon
        while not stop_condition(self):
            self.curr_episode += 1
            total_steps += self.run_ep()
            self.epsilon = max(self.min_eps,
                               self.max_eps * ((self.threshold_eps - self.curr_episode) / self.threshold_eps))
        return (total_steps, start_ep, self.epsilon)

    def run_fixed_eps(self, num_eps=150):
        eps = 0

        def ep_counter(_):
            nonlocal eps
            eps += 1
            return eps >= num_eps

        return self.run_until(ep_counter)

    def __clean_rewards(self):
        self.ep_rewards = 0.0

    def end_ep(self):
        if self.verbose:
            print("Ended episode %d with reward %d in %d steps" %
                  (self.eps, self.ep_rewards, self.ep_steps))

        self.eps += 1
        self.steps = 0
        self.__clean_rewards()

    def format_solution(self):
        from prettytable import PrettyTable, ALL
        fmt_str = "Ran Q-learning for %d episodes w/ gamma = %f, alpha = %f" % (
            self.eps, self.discount, self.learn_rate)

        table = PrettyTable([])
        table.hrules = ALL
        table.add_column("Time", ["present", "future", "total"])
        for action in self.world.actions:
            table.add_column(
                action, [self.present[action], self.future[action], self.total[action]])

        fmt_str = "%s\n\nValues:\n%s\n\nMaxQ:\n%s\n\nPolicy:\n%s" % (
            fmt_str, table, self.max_qs, self.policy)

        return fmt_str


class QPolicy(object):
    def __init__(self, q_learn):
        self.__learner = q_learn


def plot_data(data_dict, plots_dir_path):
    for x in data_dict:
        title = x['title']
        data = x['data']
        if len(data) == 1:
            plt.scatter([0], data)
        else:
            plt.plot(data)
        plt.grid(True)
        plt.title(title)
        plt.ylabel(x['y_label'])
        plt.xlabel(x['x_label'])
        plt.savefig(os.path.join(plots_dir_path, title + ".png"))
        plt.clf()

    # print('Plot Saved! - ' + plots_dir_path)


def verbose_data_dict(loss_data, eps_data):
    data_dict = []
    if loss_data is not None and len(loss_data) > 0:
        data_dict.append({'title': "Loss_vs_Epoch", 'data': loss_data,
                          'y_label': 'Loss' + '( min: ' + str(min(loss_data)) + ' )', 'x_label': 'Epoch'})
    if eps_data is not None and len(eps_data) > 0:
        data_dict.append({'title': "Eps_vs_Epoch", 'data': eps_data,
                          'y_label': 'Loss' + '( min: ' + str(min(eps_data)) + ' )', 'x_label': 'Epoch'})
    return data_dict
