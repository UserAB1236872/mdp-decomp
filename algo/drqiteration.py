import numpy as np
from ._base import _BasePlanner


class DRQIteration(_BasePlanner):
    """ Q Iteration for Decomposed Rewards """

    def __init__(self, env, discount, threshold=0.001):
        super().__init__(env, discount, threshold)

    def train(self):
        delta = float('inf')
        while delta >= self.threshold:
            delta = 0
            for state in self.state_space:
                for action in range(self.actions):
                    reward, reward_info = self.reward_fn(state)
                    for rt_i, rt in enumerate(self.reward_types):
                        q_val = self.q_values[str(state)][rt_i][action]
                        prob_val_next_state = 0
                        for next_state in self.state_space:
                            val_next_state = max(self.q_values[str(next_state)][rt_i][_] for _ in range(self.actions))
                            prob_val_next_state += self.transition_prob_fn(state, action, next_state) * val_next_state
                        self.q_values[str(state)][rt_i][action] = reward + self.discount * prob_val_next_state
                        delta = max(delta, abs(q_val - self.q_values[str(state)][rt_i][action]))
            print(delta, self.threshold)

    def act(self, state, debug=False):
        state = str(state)
        action_q_values = [sum([self.q_values[state][r_i][a] for r_i, _ in enumerate(self.reward_types)])
                           for a in range(self.actions)]
        action = int(np.argmax(action_q_values))

        if not debug:
            return action
        else:
            q_values = [self.q_values[state][r_i] for r_i in range(len(self.reward_types))]
            rdx, msx = self._get_explanation(q_values)
            info = {'msx': msx, 'rdx': rdx, 'q_values': q_values}
            return action, info
