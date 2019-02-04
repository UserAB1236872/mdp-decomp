import random
import pickle
import torch
import numpy as np
from .utils import _LinearDecay
from torch.optim import SGD


class _BasePlanner:
    """ Base Class for implementing decomposed reward Planning algorithm"""

    def __init__(self, env, discount, threshold=0.001):
        """

        :param env: instance of the environment
        :param discount: discount for future rewards
        :param threshold:
        """
        self.env = env
        self.actions = env.action_space.n
        self.state_space = env.states
        self.transition_prob_fn = env.transition_prob
        self.reward_fn = env.reward
        self.reward_types = sorted(env.reward_types)
        self.threshold = threshold
        self.discount = discount

        self.q_values = {}  # state x reward type x action
        for state in self.state_space:
            self._ensure_state_exists(state)

    def _ensure_state_exists(self, state):
        state = state.__str__()
        if state not in self.q_values:
            self.q_values[state] = {r_i: {a: 0 for a in range(self.actions)}
                                    for r_i, _ in enumerate(self.reward_types)}
        return state

    def train(self, episodes):
        """ trains the algorithm for given episodes"""
        raise NotImplementedError

    def act(self, state, debug=False):
        """ returns greedy action for the state"""
        raise NotImplementedError

    def save(self, path):
        pickle.dump(self.q_values, open(path, 'wb'))

    def restore(self, path):
        self.q_values = pickle.load(open(path, 'rb'))

    def _get_explanation(self, q_values):
        """returns rdx and msx for the given q_values"""
        msx, rdx = {}, {}
        for a in range(self.actions):
            msx[a], rdx[a] = {}, {}
            for a_dash in range(self.actions):
                if a != a_dash:
                    rdx[a][a_dash] = {}
                    msx[a][a_dash] = [[], []]  # positive, negative
                    pos_rdx, neg_rdx = [], []

                    for rt_i, rt in enumerate(self.reward_types):
                        rdx[a][a_dash][rt] = q_values[rt_i][a] - q_values[rt_i][a_dash]
                        if rdx[a][a_dash][rt] < 0:
                            neg_rdx.append((rdx[a][a_dash][rt], rt))
                            msx[a][a_dash][1].append(rt)
                        else:
                            pos_rdx.append((rdx[a][a_dash][rt], rt))

                    neg_rdx_sum = sum(v for v, rt in neg_rdx)
                    pos_rdx = sorted(pos_rdx, reverse=True)

                    if len(neg_rdx) > 0:
                        msx_sum = 0
                        for v, rt in pos_rdx:
                            msx_sum += v
                            msx[a][a_dash][0].append(rt)
                            if msx_sum > abs(neg_rdx_sum):
                                break
                    msx[a][a_dash] = tuple(msx[a][a_dash])

        return rdx, msx


class _BaseLearner:
    """ Base Class for implementing decomposed rewards algorithms"""

    def __init__(self, env, lr, discount, min_eps, max_eps, total_episodes):
        """

        :param env: instance of the environment
        :param lr:  learning rate
        :param discount: discount
        :param min_eps: minimum epsilon for exploration
        :param max_eps: maximum epsilon for exploration
        :param total_episodes: total no. of episodes for training (used for linearly decaying the exploration rate)
        """
        self.env = env
        self.actions = env.action_space.n
        self.reward_types = sorted(env.reward_types)
        self.lr = lr
        self.discount = discount
        self.linear_decay = _LinearDecay(min_eps, max_eps, total_episodes)

    def _select_action(self, state):
        """ selects epsilon greedy action for the state"""
        if self.linear_decay.eps > random.random():
            return random.choice(range(self.actions))
        else:
            return self.act(state)

    def _get_explanation(self, q_values):
        """returns rdx and msx for the given q_values"""
        msx, rdx = {}, {}
        for a in range(self.actions):
            msx[a], rdx[a] = {}, {}
            for a_dash in range(self.actions):
                if a != a_dash:
                    rdx[a][a_dash] = {}
                    msx[a][a_dash] = [[], []]  # positive, negative
                    pos_rdx, neg_rdx = [], []

                    for rt_i, rt in enumerate(self.reward_types):
                        rdx[a][a_dash][rt] = q_values[rt_i][a] - q_values[rt_i][a_dash]
                        if rdx[a][a_dash][rt] < 0:
                            neg_rdx.append((rdx[a][a_dash][rt], rt))
                            msx[a][a_dash][1].append(rt)
                        else:
                            pos_rdx.append((rdx[a][a_dash][rt], rt))

                    neg_rdx_sum = sum(v for v, rt in neg_rdx)
                    pos_rdx = sorted(pos_rdx, reverse=True)

                    if len(neg_rdx) > 0:
                        msx_sum = 0
                        for v, rt in pos_rdx:
                            msx_sum += v
                            msx[a][a_dash][0].append(rt)
                            if msx_sum > abs(neg_rdx_sum):
                                break
                    msx[a][a_dash] = tuple(msx[a][a_dash])

        return rdx, msx

    def train(self, episodes):
        """ trains the algorithm for given episodes"""
        raise NotImplementedError

    def act(self, state, debug=False):
        """ returns greedy action for the state"""
        raise NotImplementedError

    def save(self, path):
        """ save relevant properties in given path"""
        raise NotImplementedError

    def restore(self, path):
        """ save relevant properties from given path"""
        raise NotImplementedError


class _BaseTableLearner(_BaseLearner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_values = {}  # state x reward_type x action

    def _ensure_state_exists(self, state):
        state = state.__str__()
        if state not in self.q_values:
            self.q_values[state] = {r_i: {a: 0 for a in range(self.actions)}
                                    for r_i, _ in enumerate(self.reward_types)}
        return state

    def act(self, state, debug=False):
        state = self._ensure_state_exists(state)
        action_q_values = [sum([self.q_values[state][r_i][a] for r_i, _ in enumerate(self.reward_types)])
                           for a in range(self.actions)]
        action = int(np.argmax(action_q_values))
        q_values = [self.q_values[state][r_i] for r_i in range(len(self.reward_types))]

        if not debug:
            return action
        else:
            rdx, msx = self._get_explanation(q_values)
            info = {'msx': msx, 'rdx': rdx, 'q_values': q_values}
            return action, info

    def save(self, path):
        pickle.dump(self.q_values, open(path, 'wb'))

    def restore(self, path):
        self.q_values = pickle.load(open(path, 'rb'))


class _BaseDeepLearner(_BaseLearner):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model
        self.target_model = model
        self.optimizer = SGD(self.model.parameters(), lr=self.lr)

    def act(self, state, debug=False):
        """ returns greedy action"""
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = None
        for rt, _ in enumerate(self.reward_types):
            rt_q_value = self.model(state, rt)
            q_values = torch.cat((q_values, rt_q_value)) if q_values is not None else rt_q_value
        action = int(q_values.sum(0).max(0)[1].data.numpy())
        q_values = q_values.data.numpy().tolist()

        if not debug:
            return action
        else:
            rdx, msx = self._get_explanation(q_values)
            info = {'msx': msx, 'rdx': rdx, 'q_values': q_values}
            return action, info

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def restore(self, path):
        self.model.load_state_dict(torch.load(path))
