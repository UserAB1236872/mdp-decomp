def main():
    from gridworld import Gridworld
    from gridworld.q_world import QWorld
    from solvers.q_iter import QIter
    from solvers.val_iter import ValueIter
    from explorers.q_learn import QLearn
    from explorers.sarsa import Sarsa
    import numpy as np

    np.set_printoptions(precision=2)

    world = Gridworld()
    # for action in world.actions:
    #     print("Matrix @ (2,2) action", action, ":\n",
    #           world.transition_matrix((3, 4), action))
    # print(world.rewards)

    print("Rewards for world:\n")
    for (k, v) in world.rewards.items():
        print(k, ":\n")
        print(v)
    print("total:\n", world.total_reward)

    # world = QWorld(world)
    print("\n\nSolution:")
    solver = QIter(world, beta=0.9)
    solver.solve()
    # solver = QLearn(world, verbose=False)
    # solver.run_fixed_eps(num_eps=100000)
    print(solver.format_solution())


if __name__ == "__main__":
    main()
