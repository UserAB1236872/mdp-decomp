from torch import FloatTensor
from torch.autograd import Variable
import numpy as np
from util import LinearDecay
import itertools
import random
import collections
import logging
from explorers.q_learn import QLearn
from torch.optim import Adam


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
    # def __init__(self, q_world, model, nn_archetype, mem_len=1000, swap_interval=100, batch_size=32, **kwargs):
    def __init__(self, q_world, model,algo, mem_len=5000, swap_interval=100, batch_size=32, **kwargs):
        super().__init__(q_world, **kwargs)
        state_size = q_world.shape[0] * q_world.shape[1]
        # self.curr_model = model(state_size, self.world.reward_types,
        #                         self.world.actions, nn_archetype, lr=self.learn_rate)
        # self.eval_model = model(state_size, self.world.reward_types,
        #                         self.world.actions, nn_archetype, lr=self.learn_rate)
        self.curr_model = model(state_size,self.world.reward_types,self.world.actions)
        self.eval_model = model(state_size,self.world.reward_types,self.world.actions)
        self.curr_model.load_state_dict(self.curr_model.state_dict())
        self.optimizer = Adam(self.curr_model.parameters(), lr=0.01)
        self.algo = algo(state_size,self.world.reward_types,self.world.actions)
        self.memory = collections.deque(maxlen=mem_len)
        self.swap_interval = swap_interval
        self.batch_size = batch_size

    def observe(self, action, rewards, next_state):
        experience = Experience(self.world.statify(
            self.state).flatten(), action, rewards, self.world.statify(next_state).flatten())
        self.memory.append(experience)

    def end_ep(self):
        batch = random.sample(self.memory, min(
            len(self.memory), self.batch_size))

        curr_states = Variable(FloatTensor([*map(lambda x: x.s, batch)]), requires_grad=True)
        next_states = Variable(FloatTensor([*map(lambda x: x.s_p, batch)]))
        actions = [*map(lambda x: x.a, batch)]
        rewards = [*map(lambda x: x.r, batch)]

        self.algo.update(curr_states, next_states,
                               actions, rewards, self.discount, self.curr_model, self.eval_model, self.optimizer)

        if self.eps % self.swap_interval:
            if self.verbose:
                logging.info("Swapping models")
            # self.curr_model, self.eval_model = self.eval_model, self.curr_model
            curr_params = self.curr_model.state_dict()
            eval_params = self.eval_model.state_dict()
            self.eval_model.load_state_dict(curr_params)
            self.curr_model.load_state_dict(eval_params)

        if self.verbose:
            logging.info(
                "Recalculating table policy based on current model...")

        for state in self.world.states:
            qs = self.algo.eval(self.curr_model, self.world.statify(state))
            for action in self.world.actions:
                total = 0.0
                for r_type in self.world.reward_types:
                    self.q_vals[action][r_type][state] = qs[action][r_type]
                    total += qs[action][r_type]

                self.total[action][state] = total

        self.find_policy()

        super().end_ep()
