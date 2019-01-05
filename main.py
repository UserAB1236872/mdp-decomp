import argparse
import os
import gym
import torch
import monitor
import torch.nn as nn
from algo import DRQLearn, DRSarsa


class DRModel(nn.Module):
    def __init__(self, state_size, reward_types, actions):
        super(DRModel, self).__init__()
        self.actions = {action: i for (i, action) in enumerate(actions)}
        self.reward_types = reward_types
        self.state_size = state_size

        for rt in reward_types:
            model = nn.Linear(state_size, len(actions), bias=False)
            setattr(self, 'model_{}'.format(rt), model)
            getattr(self, 'model_{}'.format(rt)).weight.data.fill_(0)

    def forward(self, input, r_type):
        return getattr(self, 'model_{}'.format(r_type))(input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--env', default='grid-world', help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--min_eps', type=float, default=0.1, help='Min Epsilon for Exploration')
    parser.add_argument('--max_eps', type=float, default=0.9, help='Max. Epislon for Exploration')
    parser.add_argument('--total_episodes', type=int, default=10000, help='Total Number of episodes for training')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation/interval')
    parser.add_argument('--train_interval', type=int, default=100, help='No. of Episodes per training interval')
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    # initialize environment
    env_fn = lambda: gym.make(args.env)

    # initialize solvers
    q_solver = DRQLearn(env_fn(), args.lr, args.discount, args.min_eps, args.max_eps, args.total_episodes)
    sarsa_solver = DRSarsa(env_fn(), args.lr, args.discount, args.min_eps, args.max_eps, args.total_episodes)
    solvers = [q_solver, sarsa_solver]

    # Fire it up!
    monitor.run(env_fn(), solvers, args.total_episodes, args.eval_episodes,
                args.train_interval, result_path=args.result_dir)
