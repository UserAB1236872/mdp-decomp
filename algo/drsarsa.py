from ._base import _BaseTableLearner


class DRSarsa(_BaseTableLearner):
    """ Sarsa for Decomposed Rewards"""

    def __init__(self, env, lr, discount, min_eps, max_eps, total_episodes, max_episode_steps=100000,
                 use_decomposition=True):
        super().__init__(env, lr, discount, min_eps, max_eps, total_episodes, max_episode_steps, use_decomposition)

    def _update(self, state, action, next_state, next_state_action, reward, done):
        for rt, r in enumerate(reward):
            target = r
            if not done:
                target += self.q_values[next_state][rt][next_state_action]

            self.q_values[state][rt][action] *= (1 - self.lr)
            self.q_values[state][rt][action] += self.lr * self.discount * target

    def train(self, episodes):
        for ep in range(episodes):
            done = False
            state = self.env.reset()
            state = self._ensure_state_exists(state)
            action = self._select_action(state)
            while not done:
                next_state, _, done, info = self.env.step(action)
                reward = [info['reward_decomposition'][k] for k in sorted(info['reward_decomposition'].keys())]
                next_state = self._ensure_state_exists(next_state)
                next_action = self._select_action(next_state)
                self._update(state, action, next_state, next_action, reward, done)
                state = next_state
                action = next_action

            self.linear_decay.update()
