import random
import itertools


class QWorld(object):
    def __init__(self, world):
        self.__world = world
        self.shape = world.shape
        self.actions = world.actions
        self.reward_types = [*world.rewards.keys()]

    def reset(self):
        choice = (random.randint(
            0, self.shape[0] - 1), random.randint(0, self.shape[1] - 1))
        while self.__world.terminal[choice]:
            choice = (random.randint(
                0, self.__world.shape[0] - 1), random.randint(0, self.__world.shape[1] - 1))

        return choice

    def act(self, state, action):
        cum_prob = 0.0
        roll = random.random()

        for (r_sp, c_sp) in itertools.product(range(self.shape[0]), range(self.shape[1])):
            cum_prob += self.__world.transition_prob(
                state, action, (r_sp, c_sp))
            if roll < cum_prob:
                rewards = {}
                for (typ, vals) in self.__world.rewards.items():
                    rewards[typ] = vals[r_sp, c_sp]
                total = self.__world.total_reward[r_sp, c_sp]

                terminal = self.__world.terminal[r_sp, c_sp]

                return (r_sp, c_sp), rewards, total, terminal

        raise Exception("No successor state")
