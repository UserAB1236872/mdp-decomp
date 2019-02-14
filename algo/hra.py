import torch
from .utils import ReplayMemory, Transition
from torch.nn import MSELoss
from torch.autograd import Variable
from ._base import _BaseDeepLearner


class HRA(_BaseDeepLearner):
    """ Hybrid Reward Architecture
        Reference: https://arxiv.org/pdf/1706.04208.pdf
    """

    def __init__(self, env, model, lr, discount, mem_len, batch_size, min_eps, max_eps, total_episodes,
                 update_target_interval=1, use_cuda=False):
        super().__init__(model, update_target_interval, use_cuda, env, lr, discount, min_eps, max_eps, total_episodes)

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
                target_q[non_final_mask] = self.model(non_final_next_state_batch, rt).max(1)[0].detach().unsqueeze(1)
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
                action = self._select_action(state)
                next_state, _, done, info = self.env.step(action)
                reward = [info['reward_decomposition'][rt] for rt in self.reward_types]
                self.update(state, action, next_state, reward, done)
                state = next_state

            self.linear_decay.update()

            if ep % self.update_target_interval == 0:
                self.update_target_model()
