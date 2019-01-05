import numpy as np
import random
from . import _LinearDecay


class QLearn():
    """ Q-Learning for Decomposed Rewards"""

    def __init__(self, actions, reward_types, lr, discount, min_eps, max_eps, total_episodes):
        """

        :param actions: Count of available actions
        :param reward_types: Count of available reward types
        :param lr: learning rate (remains constant through training)
        :param discount: discount rate
        :param min_eps: Minimum epsilon rate for exploration
        :param max_eps: Maximum epsilon rate for exploration
        :param total_episodes: Total Number of episodes for which training would be executed.
                               This is used to decay exploration rate.
        """
        self.q_values = {}
        self.actions = actions
        self.reward_types = reward_types
        self.lr = lr
        self.discount = discount
        self.linear_decay = _LinearDecay(min_eps, max_eps, total_episodes)

    def update(self, state, action, next_state, reward, done):
        if state not in self.q_values:
            self.q_values[state] = {a: {r: 0 for r in range(self.reward_types)} for a in range(self.actions)}
        if next_state not in self.q_values:
            self.q_values[state] = {a: {r: 0 for r in range(self.reward_types)} for a in range(self.actions)}

        next_state_action = self.act(next_state)
        for r_i, r in enumerate(reward):
            target = r
            if not done:
                target += self.q_values[next_state][next_state_action][r_i]

            self.q_values[state][action][r_i] *= (1 - self.lr)
            self.q_values[state][action][r_i] += self.lr * self.discount * target

    def train(self, episodes, seed):
        env = self.env_fn()
        env.seed(seed)
        for ep in range(episodes):
            done = False
            state = env.reset()
            while not done:
                if self.linear_decay.eps > random.random():
                    action = random.choice(range(self.actions))
                else:
                    action = self.act(state)

                next_state, reward, done, info = env.step(action)

                self.update(state, action, next_state, reward, done)
                state = next_state

            self.linear_decay.update()

    def act(self, state):
        """ returns greedy action"""
        return np.argmax(sum([self.q_values[state][a][r_i] for r_i in range(self.reward_types)])
                         for a in range(self.actions))
