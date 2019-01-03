from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from torch.nn import Module, MSELoss, Linear
from torch.optim import Adam
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import FloatTensor
import os
import matplotlib as mpl
import torch.nn as nn

mpl.use('Agg')  # to plot graphs over a server.
import matplotlib.pyplot as plt


class DQNModel(ABC):
    @abstractmethod
    def update(self, curr_states, next_states,
               actions, rewards, eval_model):
        raise NotImplementedError

    @abstractmethod
    def eval(self, state):
        raise NotImplementedError


# class DQNLinear(Module):
#     def __init__(self, num_inputs, num_outputs):
#         super().__init__()
#         self.layer1 = Linear(num_inputs, 10)
#         self.layer2 = Linear(10, num_outputs)
#         # self.layer1 = Linear(num_inputs, num_outputs)
#
#     def forward(self, x):
#         out = self.layer2(F.relu(self.layer1(x))
#         # out = self.layer1(x)
#         # out = self.layer2(out)
#         return out


# Sepreate Network for each reward type

class DecompDQNModel(nn.Module):
    def __init__(self, state_size, reward_types, actions):
        super(DecompDQNModel, self).__init__()
        self.actions = actions
        self.reward_types = reward_types
        self.state_size = state_size

        for r_type in reward_types:
            model = nn.Linear(state_size, len(actions), bias=False)
            setattr(self, 'model_{}'.format(r_type), model)
            getattr(self, 'model_{}'.format(r_type)).weight.data.fill_(0)

        for action in actions:
            for r_type in reward_types:
                model = nn.Linear(1, 1, bias=False)
                setattr(self, 'q_{}_{}'.format(action, r_type), model)
                getattr(self, 'q_{}_{}'.format(action, r_type)).weight.data.fill_(-5)

    def forward(self, input, r_type=None):
        if r_type is None:
            decomp_q, q_overall = None, None
            for r_type in self.reward_types:
                q_val = getattr(self, 'model_{}'.format(r_type))(input).unsqueeze(1)
                decomp_q = q_val if decomp_q is None else torch.cat((decomp_q, q_val), dim=1)
            ones = torch.ones(decomp_q.shape, requires_grad=input.requires_grad)
            weighted_ones = torch.ones(decomp_q.shape)
            for act_i, action in enumerate(self.actions):
                for r_i,r_type in enumerate(self.reward_types):
                    weighted_ones[:,r_i,[act_i]] = torch.sigmoid(getattr(self, 'q_{}_{}'.format(action, r_type))(ones[:,r_i,[act_i]]))
            q_overall = decomp_q.detach() * weighted_ones
            q_overall = q_overall.sum(1)
            return decomp_q, q_overall
        else:
            return getattr(self, 'model_{}'.format(r_type))(input)


class QModel(nn.Module):
    def __init__(self, reward_types, actions):
        super(QModel, self).__init__()
        self.actions = actions
        for action in actions:
            model = nn.Linear(len(reward_types), 1, bias=False)
            setattr(self, 'model_{}'.format(action), model)
            getattr(self, 'model_{}'.format(action)).weight.data.fill_(0)

    def forward(self, input):
        out = None
        for action in self.actions:
            q_val = getattr(self, 'model_{}'.format(action))(input)
            out = q_val if out is None else torch.cat((out, q_val))
        return out

class RewardMajorModel(DQNModel):
    def __init__(self, state_size, reward_types, actions):
        super().__init__()
        self.actions = {action: i for (i, action) in enumerate(actions)}
        self.reward_types = reward_types
        self.state_size = state_size

        self.looses = []

    def update(self, curr_states, next_states, actions, rewards, terminals, discount, curr_model, eval_model,
               optimizer):
        non_final_mask = 1 - torch.ByteTensor(terminals)
        curr_states = Variable(curr_states, requires_grad=True)
        next_states = Variable(next_states, requires_grad=False)
        reward_batch = torch.FloatTensor(rewards)

        actions = torch.LongTensor(actions).unsqueeze(1)
        repeated_actions = actions.repeat(1, len(self.reward_types)).unsqueeze(2)

        # make prediction for decomposed q-values for current_state
        decomp_q_curr_state, q_curr_state = curr_model(curr_states)
        decomp_q_curr_state = decomp_q_curr_state.gather(2, repeated_actions)
        q_curr_state = q_curr_state.gather(1, actions)

        # make prediction for decomposed q-values for next_state
        decomp_q_next_state, q_next_state = curr_model(next_states[non_final_mask])
        q_next_state, q_next_state_actions = q_next_state.max(1)
        q_next_state_actions = q_next_state_actions.unsqueeze(1).repeat(1, len(self.reward_types)).unsqueeze(2)

        # Calculate Targets
        target_q = Variable(torch.zeros(q_curr_state.shape), requires_grad=False)
        target_q[non_final_mask] = q_next_state.detach().unsqueeze(1)
        target_q = discount * (reward_batch.sum(1, keepdim=True) + target_q)

        target_decomp_q = Variable(torch.zeros(decomp_q_curr_state.shape), requires_grad=False)
        target_decomp_q[non_final_mask] = decomp_q_next_state.gather(2, q_next_state_actions)
        target_decomp_q = discount * (reward_batch.unsqueeze(2) + target_decomp_q)

        # calculate loss
        loss = MSELoss()(q_curr_state, target_q)
        loss += MSELoss()(decomp_q_curr_state, target_decomp_q)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_model.parameters(), 100)
        optimizer.step()

    def eval(self, model, state):
        out = {}
        for r_type in self.reward_types:
            pred = model(Variable(FloatTensor(
                state.flatten()).unsqueeze(0)), r_type).cpu().data[0].numpy()
            for action, i in self.actions.items():
                if action not in out:
                    out[action] = {}
                out[action][r_type] = pred[i]

        return out


def plot_data(data_dict, plots_dir_path):
    for x in data_dict:
        title = x['title']
        data = x['data']
        if len(data) == 1:
            plt.scatter([0], data)
        else:
            plt.plot(data)
        plt.grid(True)
        plt.title(title)
        plt.ylabel(x['y_label'])
        plt.xlabel(x['x_label'])
        plt.savefig(os.path.join(plots_dir_path, title + ".png"))
        plt.clf()

    # print('Plot Saved! - ' + plots_dir_path)


def verbose_data_dict(dqn_looses):
    data_dict = []
    if dqn_looses is not None and len(dqn_looses) > 0:
        data_dict.append({'title': "Loss_vs_Epoch", 'data': dqn_looses,
                          'y_label': 'Loss' + '( min: ' + str(min(dqn_looses)) + ' )', 'x_label': 'Epoch'})
    return data_dict
