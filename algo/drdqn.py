import torch
from .utils import ReplayMemory, Transition
from torch.nn import MSELoss
from torch.autograd import Variable
from ._base import _BaseDeepLearner

import logging
import pickle


class DRDQN(_BaseDeepLearner):
    """ DQN for Decomposed Rewards"""

    def __init__(self, env, model, lr, discount, mem_len, batch_size, min_eps, max_eps, total_episodes):
        super().__init__(model, env, lr, discount, min_eps, max_eps, total_episodes)

        self.batch_size = batch_size
        self.memory = ReplayMemory(mem_len)

    def update(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

        if len(self.memory) >= self.batch_size:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            state_batch = torch.FloatTensor(list(batch.state))
            action_batch = torch.LongTensor(list(batch.action)).unsqueeze(1)
            reward_batch = torch.FloatTensor(list(batch.reward))
            next_state_batch = torch.FloatTensor(list(batch.next_state))
            non_final_mask = 1 - torch.ByteTensor(list(batch.done))
            non_final_next_state_batch = next_state_batch[non_final_mask]

            # determine optimal action for next states
            policy_next_qs = None
            for rt, _ in enumerate(self.reward_types):
                next_r_qs = self.model(non_final_next_state_batch, rt)
                policy_next_qs = next_r_qs + \
                    (policy_next_qs if policy_next_qs is not None else 0)
            policy_next_actions = policy_next_qs.max(1)[1].unsqueeze(1)

            # calculate loss for each reward type
            loss = 0
            for rt, _ in enumerate(self.reward_types):
                state = Variable(state_batch.data.clone(), requires_grad=True)
                predicted_q = self.model(state, rt).gather(1, action_batch)
                reward = reward_batch[:, rt].unsqueeze(1)
                target_q = Variable(torch.zeros(
                    predicted_q.shape), requires_grad=False)
                target_q[non_final_mask] = self.model(
                    non_final_next_state_batch, rt).gather(1, policy_next_actions)
                target_q = reward + self.discount * target_q
                loss += MSELoss()(predicted_q, target_q)

            # update the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
            self.optimizer.step()

    def save(self, path, train_checkpoint=False):
        """
        Save the things we need to resume training
        """
        if train_checkpoint:
            pickle.dump(self.memory, open(path+".mem", 'wb'))
        super().save(path, train_checkpoint=train_checkpoint)

    def restore(self, path, train_checkpoint=False):
        if train_checkpoint:
            self.mem = pickle.load(open(path+".mem", 'rb'))
        super().restore(path, train_checkpoint=train_checkpoint)
