from torch import Tensor
import numpy as np
from util import LinearDecay
import itertools
import random
import collections
import logging
from explorers.q_learn import QLearn


class Experience(object):
    def __init__(self, s, a, r, s_p):
        self.__s = s
        self.__a = a
        self.__r = r
        self.__s_p = s_p

    @property
    def s(self):
        return self.__s

    @property
    def a(self):
        return self.__a

    @property
    def r(self):
        return self.__r

    @property
    def s_p(self):
        return self.__s_p


class DQN(QLearn):
    def __init__(self, q_world, model, mem_len=500000, swap_interval=100, batch_size=10, **kwargs):
        super().__init__(q_world, **kwargs)
        self.curr_model = model()
        self.eval_model = model()

        self.memory = collections.deque(maxlen=mem_len)
        self.swap_interval = swap_interval
        self.batch_size = batch_size

    def observe(self, action, rewards, next_state):
        experience = Experience(self.world.statify(
            self.state), action, rewards, self.world.statify(next_state))
        self.memory.append(experience)

    def end_ep(self):
        batch = random.sample(self.memory, min(
            len(self.memory), self.batch_size))

        curr_states = Tensor([*map(lambda x: x.s, batch)])
        next_states = Tensor([*map(lambda x: x.s_p, batch)])
        actions = [*map(lambda x: x.a, batch)]
        rewards = [*map(lambda x: x.r, batch)]

        self.curr_model.update(curr_states, next_states,
                               actions, rewards, self.eval_model)

        if self.eps % self.swap_interval:
            if self.verbose:
                logging.info("Swapping models")
            self.curr_model, self.eval_model = self.eval_model, self.curr_model

        if self.verbose:
            logging.info(
                "Recalculating table policy based on current model...")

        for state in self.world.states:
            qs = self.curr_model.eval(self.world.statify(state))
            for action in self.world.actions:
                total = 0.0
                for r_type in self.world.reward_types:
                    self.q_vals[action][r_type][state] = qs[action][r_type]
                    total += qs[action][r_type]

                self.total[action][state] = total

        self.find_policy()

        super().end_ep()
