from gridworld.general import Gridworld
import numpy as np


class MiniGridworld(Gridworld):
    def __init__(self, misfire_prob=0.1):
        rewards = {
            "success": np.array([
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 0]]),
            "fail": np.array([
                [0, 0],
                [0, 0],
                [0, 0],
                [0, -1]
            ])
        }

        terminals = np.array([
            [False, False],
            [False, False],
            [False, True],
            [False, True]
        ])

        misfires = np.array([
            ['', ''],
            ['', ''],
            ['', ''],
            ['r', ''],
        ])

        impassable = np.array([
            [True, True],
            [False, False],
            [False, False],
            [False, False],
        ])

        super().__init__(rewards, terminals, misfires,
                         impassable, terminals.shape, misfire_prob=misfire_prob)


def test_create_gridworld():
    gridworld = MiniGridworld()
    assert(gridworld.rewards["success"][2, 1] == 1)
