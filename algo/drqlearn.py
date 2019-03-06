from ._base import _BaseTableLearner


class DRQLearn(_BaseTableLearner):
    """ Q-Learning for Decomposed Rewards"""

    def __init__(self, env, lr, discount, min_eps, max_eps, total_episodes, max_episode_steps=100000,
                 use_decomposition=True):
        super().__init__(env, lr, discount, min_eps, max_eps, total_episodes, max_episode_steps, use_decomposition)

    @property
    def __name__(self):
        return 'drQ learning' if self.use_decomposition else 'Q learning'

    def _update(self, state, action, next_state, reward, done):
        next_state_action = self.act(next_state)
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
            while not done:
                action = self._select_action(state)
                next_state, _, done, info = self.env.step(action)
                reward = [info['reward_decomposition'][rt] for rt in self.reward_types]
                next_state = self._ensure_state_exists(next_state)
                self._update(state, action, next_state, reward, done)
                state = next_state

            self.linear_decay.update()
