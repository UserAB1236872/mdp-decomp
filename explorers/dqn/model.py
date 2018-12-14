from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from torch.nn import Module, MSELoss, Linear
from torch.optim import Adam
from torch.autograd import Variable
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


class DQNLinear(Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        # self.layer1 = Linear(num_inputs, 10)
        # self.layer2 = Linear(10, num_outputs)
        self.layer1 = Linear(num_inputs, num_outputs)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer1(x)
        # out = self.layer2(out)
        return out


class DecompDQNModel(nn.Module):
    def __init__(self, state_size, reward_types, actions):
        super(DecompDQNModel, self).__init__()
        self.actions = {action: i for (i, action) in enumerate(actions)}
        self.reward_types = reward_types
        self.state_size = state_size

        # self.models = {}

        for r_type in reward_types:
            model = nn.Sequential(nn.Linear(state_size, len(actions)))
            setattr(self, 'model_{}'.format(r_type), model)

    def forward(self, input, r_type):
        return getattr(self, 'model_{}'.format(r_type))(input)


class RewardMajorModel(DQNModel):
    def __init__(self, state_size, reward_types, actions):
        super().__init__()
        self.actions = {action: i for (i, action) in enumerate(actions)}
        self.reward_types = reward_types
        self.state_size = state_size

        # self.models = {}

        for r_type in reward_types:
            model = nn.Sequential(nn.Linear(state_size, len(actions)))
            setattr(self, 'model_{}'.format(r_type), model)

        # for r_type in reward_types:
        #     self.models[r_type] = {}
        #     self.models[r_type]['nn'] = nn_archetype(state_size, len(actions))
        #     self.models[r_type]['optimizer'] = Adam(
        #         self.models[r_type]['nn'].parameters(), lr=lr)

        self.looses = []

    def update(self, curr_states, next_states,
               actions, rewards, discount, curr_model, eval_model, optimizer):
        reward_major = {}
        # batch_size = len(actions)
        for r_map in rewards:
            for r_type, obs in r_map.items():
                if r_type not in reward_major:
                    reward_major[r_type] = []
                reward_major[r_type].append(obs)

        r_type_qs = {}
        policy_qs, policy_next_qs = None, None
        for r_type in self.reward_types:
            input = Variable(curr_states.data.clone(), requires_grad=True)
            r_type_qs[r_type] = curr_model(input, r_type)
            policy_qs = r_type_qs[r_type] + (policy_qs if policy_qs is not None else 0)
            policy_next_qs = curr_model(next_states, r_type) + (policy_next_qs if policy_next_qs is not None else 0)
        # for r_type, model in self.models.items():
        #     r_type_qs[r_type] = model['nn'](Variable(curr_states.data.clone(), requires_grad=True))
        #     policy_qs = r_type_qs[r_type] + (policy_qs if policy_qs is not None else 0)
        #     policy_next_qs = model['nn'](next_states) + (policy_next_qs if policy_next_qs is not None else 0)

        _, policy_actions = policy_qs.max(1)
        policy_next_qs, policy_next_actions = policy_next_qs.max(1)
        optimizer.zero_grad()
        loss = 0

        for r_type in self.reward_types:
            # typed_eval_model = getattr(eval_model, 'model_{}'.format(r_type))
            reward_batch = FloatTensor(reward_major[r_type])
            target_q = Variable(
                r_type_qs[r_type].data.clone(), requires_grad=False)
            eval_next_qs = eval_model(next_states,r_type).data
            for i, rwd in enumerate(reward_batch):
                target_q[i, policy_actions[i].data[0]] = rwd + discount * \
                                                         eval_next_qs[i, policy_next_actions[i].data[0]]

            loss += MSELoss()(r_type_qs[r_type], target_q)
        loss.backward()
        optimizer.step()
        self.looses.append(loss.data.item())
        # if len(self.looses) % 100 == 0:
        #     plot_data(verbose_data_dict(self.looses), os.getcwd())

    def eval(self,model, state):
        out = {}
        for r_type in self.reward_types:
            pred = model(Variable(FloatTensor(
                state.flatten()).unsqueeze(0)),r_type).cpu().data[0].numpy()
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
