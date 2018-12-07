from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from torch.nn import Module, MSELoss
from torch import Tensor
import torch


class DQNModuleArchetype(Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, num_inputs, num_outputs):
        raise NotImplementedError


class DQNModel(ABC):
    @abstractmethod
    def update(self, curr_states, next_states,
               actions, rewards, eval_model):
        raise NotImplementedError

    @abstractmethod
    def eval(self, state):
        raise NotImplementedError


class RewardMajorModel(DQNModel):
    def __init__(self, reward_types, actions, nn_archetype):
        self.actions = {action: i for (i, action) in enumerate(actions)}
        self.reward_types = reward_types

        self.models = {}

        for r_type in reward_types:
            self.models[r_type] = nn_archetype(len(actions))

    def update(self, curr_states, next_states,
               actions, rewards, discount, eval_model):
        reward_major = {}
        batch_size = len(actions)
        for r_map in rewards:
            for r_type, obs in r_map.items():
                if r_type not in reward_major:
                    reward_major[r_type] = []
                reward_major[r_type].append(obs)

        policy_qs = torch.zeros(batch_size)
        policy_next_qs = torch.zeros(batch_size)
        for model in self.models.values():
            policy_qs += model(curr_states)
            policy_next_qs += model(next_states)

        policy_qs, policy_actions = policy_qs.max(0)
        _, policy_next_actions = policy_next_qs.max(0)

        for r_type, model in self.models.items():
            typed_eval_model = eval_model.models[r_type]
            reward_batch = Tensor(reward_major[r_type])
            q = reward_batch + discount * \
                typed_eval_model(next_states)[:, policy_next_actions]
            loss = MSELoss(policy_qs[:, policy_actions], q)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

    def eval(self, state):
        out = {}
        for r_type, model in self.models.items():
            pred = model(state).detach()
            for action, i in self.actions.items():
                if action not in out:
                    out[action] = {}
                out[action][r_type] = pred[i].numpy()

        return out
