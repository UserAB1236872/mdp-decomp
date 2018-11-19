def main():
    from gridworld.worlds import CliffWorld
    from gridworld.q_world import QWorld
    from solvers.q_iter import QIter
    from solvers.val_iter import ValueIter
    from explorers.q_learn import QLearn
    from explorers.sarsa import Sarsa
    import numpy as np

    np.set_printoptions(precision=2)

    world = CliffWorld()
    # for action in world.actions:
    #     print("Matrix @ (2,2) action", action, ":\n",
    #           world.transition_matrix((3, 4), action))
    # print(world.rewards)

    print("World:\n")
    print(world.printable())

    # Run for testing purposes
    solver = ValueIter(world)
    solver.solve

    print("\n\nExact Solution:")
    solver = QIter(world, beta=0.9)
    solver.solve()

    print(solver.format_solution())

    print("\n\n QLearning:")
    world = QWorld(world)
    solver = QLearn(world, verbose=False)
    solver.run_fixed_eps(num_eps=100000)
    print(solver.format_solution())

    print("\n\n Sarsa:")
    solver = Sarsa(world, verbose=False)
    solver.run_fixed_eps(num_eps=100000)
    print(solver.format_solution())


if __name__ == "__main__":
    main()
