import numpy as np
import random
from .utils import _LinearDecay
import pickle


class DRSarsa:
    """ Sarsa for Decomposed Rewards"""

    def __init__(self, env, lr, discount, min_eps, max_eps, total_episodes, eps_max_episodes):
        self.env = env
        self.actions = env.action_space.n
        self.reward_types = len(env.reward_types)
        self.eps_max_episodes = eps_max_episodes
        self.lr = lr
        self.discount = discount
        self.linear_decay = _LinearDecay(min_eps, max_eps, total_episodes)

        self.q_values = {}

    def _ensure_state_exists(self, state):
        state = state.__str__()
        if state not in self.q_values:
            self.q_values[state] = {a: {r: 0 for r in range(self.reward_types)} for a in range(self.actions)}
        return state

    def update(self, state, action, next_state, next_state_action, reward, done):
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
            state = self._ensure_state_exists(state)
            while not done:
                action = self._select_action(state)
                next_state, _, done, info = self.env.step(action)
                reward = [info['reward_decomposition'][k] for k in sorted(info['reward_decomposition'].keys())]
                next_state = self._ensure_state_exists(next_state)
                next_action = self._select_action(next_state)
                self.update(state, action, next_state, next_action, reward, done)
                state = next_state

            self.linear_decay.update()

    def act(self, state, debug=False):
        state = self._ensure_state_exists(state)
        q_values = [sum([self.q_values[state][a][r_i] for r_i in range(self.reward_types)])
                    for a in range(self.actions)]
        action = int(np.argmax(q_values))

        if not debug:
            return action
        else:
            return action, self.q_values[state]

    def save(self, path):
        pickle.dump(self.q_values, open(path, 'wb'))

    def restore(self, path):
        self.q_values = pickle.load(open(path, 'rb'))
