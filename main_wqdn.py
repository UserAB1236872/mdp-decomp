def main():
    from gridworld.worlds import CliffWorld
    from solvers.q_iter import QIter
    from explorers.q_learn import QLearn
    from explorers.sarsa import Sarsa
    from explorers.wdqn import WDQN
    from explorers.wdqn.model import RewardMajorModel, DecompDQNModel
    from monitors.closeness import SolutionMonitor
    import numpy as np
    import logging
    logging.getLogger().setLevel(logging.INFO)

    np.set_printoptions(floatmode='fixed', precision=2)

    world = CliffWorld()

    def make_dqn(w, verbose=True, max_episodes=None):
        return WDQN(w, DecompDQNModel, RewardMajorModel, verbose=verbose, max_episodes=max_episodes)

    explorers = [make_dqn]
    exact = QIter

    monitor = SolutionMonitor(world, exact, explorers, max_steps=20000)
    monitor.multi_run()


if __name__ == "__main__":
    main()
