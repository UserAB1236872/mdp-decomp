def main():
    from gridworld.worlds import CliffWorld
    from solvers.q_iter import QIter
    from explorers.q_learn import QLearn
    from explorers.sarsa import Sarsa
    from explorers.dqn import DQN
    from explorers.dqn.model import RewardMajorModel, DecompDQNModel
    from monitors.closeness import SolutionMonitor
    import numpy as np
    import logging
    logging.getLogger().setLevel(logging.INFO)

    np.set_printoptions(floatmode='fixed', precision=2)

    world = CliffWorld()

    # def make_dqn(w, verbose=True):
    #     return DQN(w, RewardMajorModel, DQNLinear, verbose=verbose)
    def make_dqn(w, verbose=True, max_episodes=None):
        return DQN(w, DecompDQNModel, RewardMajorModel, verbose=verbose, max_episodes=max_episodes)

    explorers = [QLearn, Sarsa, make_dqn]
    # explorers = [QLearn]
    # explorers = [make_dqn]
    exact = QIter

    monitor = SolutionMonitor(world, exact, explorers, max_steps=10000)
    monitor.multi_run()


if __name__ == "__main__":
    main()
