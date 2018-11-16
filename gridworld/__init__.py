class Gridworld(object):
    def __init__(self):
        import numpy as np
        cliff = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, -10, -10, -10, 0],
            [0, 0, 0, 0, 0]
        ])

        success = np.array([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 15],
            [0, 0, 0, 0, 0]
        ])

        fail = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        gold = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 15, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        self.rewards = {
            "cliff": cliff,
            "success": success,
            "fail": fail,
            "gold": gold,
        }

        self.total_reward = np.zeros((4, 5))
        for (_, v) in self.rewards.items():
            self.total_reward += v

        self.terminal = np.array([
            [False, False, False, False, True],
            [False, False, False, False, True],
            [False, True,  True,  True,  True],
            [False, False, False, False, False]
        ])

        self.shape = (4, 5)

        self.actions = ['u', 'd', 'l', 'r']

        self.states = []

        for r in range(0, 4):
            for c in range(0, 6):
                self.states.append((r, c))

    def transition_prob(self, state, action, nxt):
        if state[0] < 0 or state[0] > 3 or state[1] < 0 or state[1] > 4:
            # Out of bounds but this makes code prettier
            return 0.0

        if nxt[0] < 0 or nxt[0] > 3 or nxt[1] < 0 or nxt[1] > 4:
            # Out of bounds but this makes code prettier
            return 0.0

        if self.terminal[state]:
            return 0.0

        # Boundary guard
        if action == 'l' and state[1] == 0:
            if state == nxt:
                return 1.0
            else:
                return 0.0

        if action == 'r' and state[1] == 4:
            if state == nxt:
                return 1.0
            else:
                return 0.0

        if action == 'u' and state[0] == 0:
            if state == nxt:
                return 1.0
            else:
                return 0.0

        if action == 'd' and state[0] == 3:
            if state == nxt:
                return 1.0
            else:
                return 0.0

        # cliff slipping
        if state[0] == 3 and (state[1] >= 1 and state[1] <= 3) and (action == 'r' or action == 'l'):
            if action == 'r' and nxt[0] == state[0] and nxt[1] == state[1] + 1:
                return 0.9
            elif action == 'l' and nxt[0] == state[0] and nxt[1] == state[1] - 1:
                return 0.9
            elif (action == 'r' or action == 'l') and nxt[0] == state[0] - 1 and nxt[1] == state[1]:
                return 0.1
            else:
                return 0.0

        # Normal dynamics
        if action == 'u':
            if nxt[0] == state[0] - 1 and nxt[1] == state[1]:
                return 1.0
            else:
                return 0.0

        if action == 'd':
            if nxt[0] == state[0] + 1 and nxt[1] == state[1]:
                return 1.0
            else:
                return 0.0

        if action == 'l':
            if nxt[0] == state[0] and nxt[1] == state[1] - 1:
                return 1.0
            else:
                return 0.0

        if action == 'r':
            if nxt[0] == state[0] and nxt[1] == state[1] + 1:
                return 1.0
            else:
                return 0.0

        return 0.0

    def transition_matrix(self, state, action):
        import itertools
        import numpy as np

        rows = self.shape[0]
        cols = self.shape[1]

        mat = np.zeros((rows, cols))
        for (r, c) in itertools.product(range(rows), range(cols)):
            mat[r, c] = self.transition_prob(state, action, (r, c))

        return mat
