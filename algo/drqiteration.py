import numpy as np
from ._base import _BaseTablePlanner


class DRQIteration(_BaseTablePlanner):
    """ Q Iteration for Decomposed Rewards """

    def __init__(self, env, discount, threshold=0.001):
        super().__init__(env, discount, threshold)

    def policy_evaluation(self, policy, verbose=False):
        delta = float('inf')
        i = 0
        while delta >= self.threshold:
            delta = 0
            for state in self.state_space:
                for action in range(self.actions):
                    reward, reward_info = self.reward_fn(state)
                    for rt_i, rt in enumerate(self.reward_types):
                        q_val = self.q_values[str(state)][rt_i][action]
                        if not self.is_terminal(state):

                            # Update Q-values using optimal action for next states
                            prob_val_next_state = 0
                            for next_state in self.state_space:
                                trans_prob = self.transition_prob_fn(state, action, next_state)
                                if trans_prob != 0:
                                    # find optimal action for next_states
                                    optimal_act = policy(next_state)
                                    val_next_state = self.q_values[str(next_state)][rt_i][optimal_act]
                                    prob_val_next_state += trans_prob * val_next_state
                            self.q_values[str(state)][rt_i][action] = reward_info[rt] + \
                                                                      self.discount * prob_val_next_state
                        else:
                            self.q_values[str(state)][rt_i][action] = reward_info[rt]

                        delta = max(delta, abs(q_val - self.q_values[str(state)][rt_i][action]))
            i += 1
            if verbose:
                print('iteration:{} delta:{}/ threshold:{}'.format(i, delta, self.threshold))

    def train(self, verbose=False):
        delta = float('inf')
        i = 0
        while delta >= self.threshold:
            delta = 0
            for state in self.state_space:
                for action in range(self.actions):
                    reward, reward_info = self.reward_fn(state)
                    for rt_i, rt in enumerate(self.reward_types):
                        q_val = self.q_values[str(state)][rt_i][action]
                        if not self.is_terminal(state):
                            # find optimal action for next_states
                            qs = []
                            for _act in range(self.actions):
                                prob_val_next_state = 0
                                for next_state in self.state_space:
                                    trans_prob = self.transition_prob_fn(state, _act, next_state)
                                    if trans_prob != 0:
                                        val_next_state = sum(self.q_values[str(next_state)][_rt_i][_act]
                                                             for _rt_i, _ in enumerate(self.reward_types))
                                        prob_val_next_state += val_next_state
                                qs.append(prob_val_next_state)
                            optimal_act = np.argmax(qs)

                            # Update Q-values using optimal action for next states
                            prob_val_next_state = 0
                            for next_state in self.state_space:
                                trans_prob = self.transition_prob_fn(state, action, next_state)
                                if trans_prob != 0:
                                    val_next_state = self.q_values[str(next_state)][rt_i][optimal_act]
                                    prob_val_next_state += trans_prob * val_next_state
                            self.q_values[str(state)][rt_i][action] = reward_info[rt] + \
                                                                      self.discount * prob_val_next_state
                        else:
                            self.q_values[str(state)][rt_i][action] = reward_info[rt]

                        delta = max(delta, abs(q_val - self.q_values[str(state)][rt_i][action]))
            i += 1
            if verbose:
                print('iteration:{} delta:{}/ threshold:{}'.format(i, delta, self.threshold))

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
