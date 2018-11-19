def main():
    from gridworld.worlds import CliffWorld
    from solvers.q_iter import QIter
    from explorers.q_learn import QLearn
    from explorers.sarsa import Sarsa
    from monitors.closeness import SolutionMonitor
    import numpy as np
    import logging
    logging.getLogger().setLevel(logging.INFO)

    np.set_printoptions(floatmode='fixed', precision=2)

    world = CliffWorld()

    explorers = [QLearn, Sarsa]
    exact = QIter

    monitor = SolutionMonitor(world, exact, explorers, max_steps=20000)
    monitor.compute(append=False)


if __name__ == "__main__":
    main()
