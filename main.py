def main():
    from gridworld.worlds import CliffWorld
    from solvers.q_iter import QIter
    from explorers.q_learn_pres_future import  QLearnPresFuture
    # from explorers.q_learn import QLearn
    from monitors.closeness import SolutionMonitor
    import numpy as np
    import logging
    logging.getLogger().setLevel(logging.INFO)

    np.set_printoptions(floatmode='fixed', precision=2)

    world = CliffWorld()

    explorers = [QLearnPresFuture]
    exact = QIter

    monitor = SolutionMonitor(world, exact, explorers, max_steps=20000)
    monitor.multi_run()


if __name__ == "__main__":
    main()
