import torch
from .utils import ReplayMemory, Transition
from torch.nn import MSELoss, SmoothL1Loss
from torch.autograd import Variable
from ._base import _BaseDeepLearner
import numpy as np


class HRA(_BaseDeepLearner):
    """ Hybrid Reward Architecture
        Reference: https://arxiv.org/pdf/1706.04208.pdf
    """

    def __init__(self, env, model, lr, discount, mem_len, batch_size, min_eps, max_eps, total_episodes,
                 update_target_interval=1, use_cuda=False, max_episode_steps=100000, use_decomposition=True):
        super().__init__(model, update_target_interval, use_cuda, env, lr, discount, min_eps, max_eps, total_episodes,
                         max_episode_steps, use_decomposition)

        self.batch_size = batch_size
        self.memory = ReplayMemory(mem_len)

    @property
    def __name__(self):
        return 'hra' if self.use_decomposition else 'unhra'

    def _update(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

        if len(self.memory) >= self.batch_size:
            self.model.train()
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            state_batch = torch.FloatTensor(list(batch.state))
            action_batch = torch.LongTensor(list(batch.action)).unsqueeze(1)
            reward_batch = torch.FloatTensor(list(batch.reward))
            next_state_batch = torch.FloatTensor(list(batch.next_state))
            non_final_mask = 1 - torch.ByteTensor(list(batch.done))
            non_final_next_state_batch = next_state_batch[non_final_mask]

            if self.use_cuda:
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                non_final_mask = non_final_mask.cuda()
                non_final_next_state_batch = non_final_next_state_batch.cuda()

            # calculate loss for each reward type
            loss = 0
            for rt, _ in enumerate(self.reward_types):
                state = Variable(state_batch.data.clone(), requires_grad=True)
                predicted_q = self.model(state, rt).gather(1, action_batch)
                reward = reward_batch[:, rt].unsqueeze(1)

                target_q = Variable(torch.zeros(predicted_q.shape), requires_grad=False)
                if self.use_cuda:
                    target_q = target_q.cuda()
                non_final_target = self.target_model(non_final_next_state_batch, rt).max(1)[0].detach().unsqueeze(1)
                target_q[non_final_mask] = non_final_target
                target_q = reward + self.discount * target_q
                loss += SmoothL1Loss()(predicted_q, target_q)

            # update the model
            loss /= len(self.reward_types)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
            self.optimizer.step()
            self.step_count += 1

            if (self.step_count % self.update_target_interval) == 0:
                self.update_target_model()
            self.model.eval()
            return loss.item()
        return None

    def train(self, episodes):
        train_loss = []
        perf = 0
        for ep in range(episodes):
            done = False
            state = self.env.reset()
            step = 0
            ep_loss = []
            while not done:
                action = self._select_action(state)
                next_state, step_reward, done, info = self.env.step(action)
                if self.use_decomposition:
                    reward = [info['reward_decomposition'][k] for k in self.reward_types]
                else:
                    reward = [step_reward]
                loss = self._update(state, action, next_state, reward, done)
                if loss is not None:
                    ep_loss.append(loss)
                perf += sum(reward)
                state = next_state
                done = done or (step > self.max_episode_steps)
                step += 1
            if len(ep_loss) > 0:
                ep_loss = np.average(ep_loss)
                train_loss.append(ep_loss)
            self.linear_decay.update()

        train_loss = np.average(train_loss)
        perf /= episodes
        return perf, train_loss, self.linear_decay.eps, self.step_count

# explore and add to experience buffer
# train n-batches
