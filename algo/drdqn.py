import numpy as np
import torch
import random
from .utils import _LinearDecay, ReplayMemory, Transition
from torch.optim import Adam
from torch.nn import MSELoss
from torch.autograd import Variable


class DRDQN:
    """ DQN for Decomposed Rewards"""

    def __init__(self, env, model, lr, discount, mem_len, batch_size, min_eps, max_eps, total_episodes):

        self.env = env
        self.actions = env.action_space.n
        self.reward_types = env.reward_types.n
        self.lr = lr
        self.discount = discount
        self.linear_decay = _LinearDecay(min_eps, max_eps, total_episodes)

        self.model = model
        self.target_model = model
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.memory = ReplayMemory(mem_len)
        self.batch_size = batch_size

    def update(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

        if len(self.memory) >= self.batch_size:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            next_state_batch = torch.cat(batch.next_state)
            non_final_mask = 1 - torch.ByteTensor(torch.cat(batch.done))
            non_final_next_state_batch = next_state_batch[non_final_mask]

            # determine optimal action for next states
            policy_next_qs = None
            for rt in range(self.reward_types):
                next_r_qs = self.model(non_final_next_state_batch, rt)
                policy_next_qs = next_r_qs + (policy_next_qs if policy_next_qs is not None else 0)
            policy_next_actions = policy_next_qs.max(1)[1].unsqueeze(1)

            # calculate loss for each reward type
            loss = 0
            for rt in range(self.reward_types):
                state = Variable(state_batch.data.clone(), requires_grad=True)
                predicted_q = self.model(state, rt).gather(1, action_batch)
                reward = torch.FloatTensor(reward_batch[rt])
                target_q = Variable(torch.zeros(predicted_q.shape), requires_grad=False)
                target_q[non_final_mask] = self.model(non_final_next_state_batch, rt).gather(1, policy_next_actions)
                target_q = reward + self.discount * target_q
                loss += MSELoss()(predicted_q, target_q)

            # update the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
            self.optimizer.step()

    def train(self, episodes):
        for ep in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                if self.linear_decay.eps > random.random():
                    action = random.choice(range(self.actions))
                else:
                    action = self.act(state)

                next_state, reward, done, info = self.env.step(action)

                self.update(state, action, next_state, reward, done)
                state = next_state

            self.linear_decay.update()

    def _select_action(self, state):
        if self.linear_decay.eps > random.random():
            return random.choice(range(self.actions))
        else:
            return self.act(state)

    def act(self, state):
        """ returns greedy action"""
        return np.argmax(sum([self.q_values[state][a][r_i] for r_i in range(self.reward_types)])
                         for a in range(self.actions))
