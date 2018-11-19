from gridworld.general import Gridworld
import numpy as np

import itertools


class MiniGridworld(Gridworld):
    def __init__(self, misfire_prob=0.1):
        rewards = {
            "success": np.array([
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 0]], dtype=float),
            "fail": np.array([
                [0, 0],
                [0, 0],
                [0, 0],
                [0, -1]
            ], dtype=float)
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


def test_gridworld():
    gridworld = MiniGridworld()
    # just make sure things work
    assert(gridworld.rewards["success"][2, 1] == 1)


def test_states():
    gridworld = MiniGridworld()

    states = set(gridworld.states)
    # Make sure you can call it twice
    assert(states == set(gridworld.states))

    shape = gridworld.shape
    states2 = set(filter(lambda x: not gridworld.impassable[x], itertools.product(
        range(shape[0]), range(shape[1]))))

    assert(states == states2)

    nonterminals = set(filter(lambda x: not gridworld.terminals[x], states2))

    assert(nonterminals == set(gridworld.nonterminal_states()))


def test_successors():
    gridworld = MiniGridworld()

    source = (1, 0)
    succs = gridworld.succ_map(source)

    # Testing impassable terrain
    assert(succs['u'] == source)
    assert(succs['d'] == (2, 0))
    assert(succs['l'] == source)
    assert(succs['r'] == (1, 1))

    succ_set = gridworld.successors(source)
    assert(len(succ_set) == 3)
    for succ in succs.values():
        assert(succ in succ_set)

    # This is a terminal state, but good for testing
    source = (2, 1)
    succs = gridworld.succ_map(source)
    assert(succs['u'] == (1, 1))
    assert(succs['r'] == source)
    assert(succs['l'] == (2, 0))
    assert(succs['d'] == (3, 1))

    succ_set = gridworld.successors(source)
    assert(len(succ_set) == 4)
    for succ in succs.values():
        assert(succ in succ_set)


def test_transition_prob():
    gridworld = MiniGridworld()

    source = (1, 0)
    nxt = (1, 1)
    assert(abs(gridworld.transition_prob(source, 'r', nxt) - 1.0) < 1e-4)
    for state in gridworld.states:
        if state == nxt:
            continue
        assert(gridworld.transition_prob(source, 'r', state) == 0)

    assert(abs(gridworld.transition_prob(source, 'u', source) - 1.0) < 1e-4)
    assert(abs(gridworld.transition_prob(source, 'l', source) - 1.0) < 1e-4)
    for state in gridworld.states:
        if state == source:
            continue
        assert(gridworld.transition_prob(source, 'u', state) == 0)
        assert(gridworld.transition_prob(source, 'l', state) == 0)

    mat = gridworld.transition_matrix(source, 'r')
    verify = np.zeros(gridworld.shape)
    verify[1, 1] = 1.0
    assert((mat == verify).all())
    mat[1, 1] = 0.0
    assert(mat.sum() == 0.0)


def test_transition_misfire_prob():
    gridworld = MiniGridworld()

    source = (3, 0)
    misfire = (3, 1)
    nxt = (2, 0)
    assert(abs(gridworld.transition_prob(source, 'u', nxt) - 0.9) < 1e-4)
    assert(abs(gridworld.transition_prob(source, 'u', misfire) - 0.1) < 1e-4)

    for state in gridworld.states:
        if state == nxt or state == misfire:
            continue
        assert(gridworld.transition_prob(source, 'u', state) == 0)

    nxt = misfire
    assert(abs(gridworld.transition_prob(source, 'r', nxt) - 1.0) < 1e-4)
    for state in gridworld.states:
        if state == nxt:
            continue
        assert(gridworld.transition_prob(source, 'r', state) == 0)

    # This is technically behavior we don't want to rely on right now, misfires
    # shouldn't fire for moving into walls, but testing this reminds us to
    # change this to test for the right thing if we change our minds
    # and disallow this in the future
    assert(abs(gridworld.transition_prob(source, 'l', source) - 0.9) < 1e-4)
    assert(abs(gridworld.transition_prob(source, 'l', misfire) - 0.1) < 1e-4)
    for state in gridworld.states:
        if state == nxt or state == source:
            continue
        assert(gridworld.transition_prob(source, 'r', state) == 0)


def test_printable_state():
    gridworld = MiniGridworld()

    verify = np.array([
        ['x', 'x'],
        [' ', ' '],
        [' ', '1.0'],
        ['r', '-1.0']
    ])

    printables = gridworld.printable()
    assert((verify == printables['total']).all())

    verify[2, 1] = ' '
    assert((verify == printables['fail']).all())

    verify[2, 1] = '1.0'
    verify[3, 1] = ' '
    assert((verify == printables['success']).all())
