from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from torch.nn import Module, MSELoss, Linear
from torch.optim import Adam
from torch.autograd import Variable
from torch import FloatTensor
import torch


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
        self.layer1 = Linear(num_inputs, 255)
        self.layer2 = Linear(255, num_outputs)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class RewardMajorModel(DQNModel):
    def __init__(self, state_size, reward_types, actions, nn_archetype, lr=0.01):
        self.actions = {action: i for (i, action) in enumerate(actions)}
        self.reward_types = reward_types
        self.state_size = state_size

        self.models = {}

        for r_type in reward_types:
            self.models[r_type] = {}
            self.models[r_type]['nn'] = nn_archetype(state_size, len(actions))
            self.models[r_type]['optimizer'] = Adam(
                self.models[r_type]['nn'].parameters(), lr=lr)

    def update(self, curr_states, next_states,
               actions, rewards, discount, eval_model):
        reward_major = {}
        batch_size = len(actions)
        for r_map in rewards:
            for r_type, obs in r_map.items():
                if r_type not in reward_major:
                    reward_major[r_type] = []
                reward_major[r_type].append(obs)

        r_type_qs = {}
        policy_qs, policy_next_qs = None, None
        for r_type, model in self.models.items():
            r_type_qs[r_type] = model['nn'](
                Variable(curr_states.data, requires_grad=True))
            policy_qs = model['nn'](curr_states) if policy_qs is None else policy_qs + \
                r_type_qs[r_type]
            policy_next_qs = model['nn'](
                next_states) if policy_next_qs is None else policy_next_qs + model['nn'](next_states)

        _, policy_actions = policy_qs.max(1)
        policy_next_qs, policy_next_actions = policy_next_qs.max(1)

        for r_type, model in self.models.items():
            typed_eval_model = eval_model.models[r_type]['nn']
            reward_batch = FloatTensor(reward_major[r_type])
            # import pdb
            # pdb.set_trace()

            target_q = Variable(
                r_type_qs[r_type].data.clone(), requires_grad=False)
            eval_next_qs = typed_eval_model(next_states).data
            for i, rwd in enumerate(reward_batch):
                target_q[i, policy_actions[i].data[0]] = rwd + discount * \
                    eval_next_qs[i, policy_next_actions[i].data[0]]
            loss = MSELoss()(r_type_qs[r_type], target_q)
            model['optimizer'].zero_grad()
            loss.backward()
            model['optimizer'].step()

    def eval(self, state):
        out = {}
        for r_type, model in self.models.items():
            pred = model['nn'](Variable(FloatTensor(
                state.flatten()).unsqueeze(0))).cpu().data[0].numpy()
            for action, i in self.actions.items():
                if action not in out:
                    out[action] = {}
                out[action][r_type] = pred[i]

        return out
