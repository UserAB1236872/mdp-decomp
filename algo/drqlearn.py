import numpy as np
import random
from .utils import _LinearDecay


class DRQLearn():
    """ Q-Learning for Decomposed Rewards"""

    def __init__(self, env, lr, discount, min_eps, max_eps, total_episodes):
        """
        :param lr: learning rate (remains constant through training)
        :param discount: discount rate
        :param min_eps: Minimum epsilon rate for exploration
        :param max_eps: Maximum epsilon rate for exploration
        :param total_episodes: Total Number of episodes for which training would be executed.
                               This is used to decay exploration rate.
        """
        self.env = env
        self.actions = env.action_space.n
        self.reward_types = env.reward_types.n
        self.lr = lr
        self.discount = discount
        self.linear_decay = _LinearDecay(min_eps, max_eps, total_episodes)

        self.q_values = {}

    def update(self, state, action, next_state, reward, done):
        for s in [state, next_state]:
            if s not in self.q_values:
                self.q_values[s] = {a: {r: 0 for r in range(self.reward_types)} for a in range(self.actions)}

        next_state_action = self.act(next_state)
        for r_i, r in enumerate(reward):
            target = r
            if not done:
                target += self.q_values[next_state][next_state_action][r_i]

            self.q_values[state][action][r_i] *= (1 - self.lr)
            self.q_values[state][action][r_i] += self.lr * self.discount * target

    def _select_action(self, state):
        if self.linear_decay.eps > random.random():
            return random.choice(range(self.actions))
        else:
            return self.act(state)

    def train(self, episodes):
        for ep in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                action = self._select_action(state)
                next_state, reward, done, info = self.env.step(action)

                self.update(state, action, next_state, reward, done)
                state = next_state

            self.linear_decay.update()

    def act(self, state):
        """ returns greedy action"""
        return np.argmax(sum([self.q_values[state][a][r_i] for r_i in range(self.reward_types)])
                         for a in range(self.actions))
