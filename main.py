import argparse
import os
import gym
import torch
import monitor
from algo import QLearn, DQN, Sarsa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--env', default='grid-world', help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage')
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    # initialize environment
    env = gym.make(args.env)

    # initialize solvers
    q_solver = QLearn(env.action_space.n, env.reward_types.n, args.lr,
                      args.discount, args.min_eps, args.max_eps, args.total_episodes)
    sarsa_solver = Sarsa()
    solvers = [q_solver, sarsa_solver]

    # Fire it up!
    monitor.run(env, solvers, result_path=args.result_dir)
