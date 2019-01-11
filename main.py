import argparse
import os
import gym, gym_decomp
import torch
import monitor
import torch.nn as nn
from algo import DRQLearn, DRSarsa, DRDSarsa, DRDQN, HRA


class DRModel(nn.Module):
    def __init__(self, state_size, reward_types, actions):
        super(DRModel, self).__init__()
        self.actions = actions
        self.reward_types = reward_types
        self.state_size = state_size

        for rt in range(reward_types):
            model = nn.Linear(state_size, actions, bias=False)
            setattr(self, 'model_{}'.format(rt), model)
            getattr(self, 'model_{}'.format(rt)).weight.data.fill_(0)

    def forward(self, input, r_type):
        return getattr(self, 'model_{}'.format(r_type))(input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--env', default='Cliffworld-v0', help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--min_eps', type=float, default=0.1, help='Min Epsilon for Exploration')
    parser.add_argument('--max_eps', type=float, default=0.9, help='Max. Epsilon for Exploration')
    parser.add_argument('--total_episodes', type=int, default=20000, help='Total Number of episodes for training')
    parser.add_argument('--eval_episodes', type=int, default=5, help='Number of episodes for evaluation/interval')
    parser.add_argument('--train_interval', type=int, default=100, help='No. of Episodes per training interval')
    parser.add_argument('--runs', type=int, default=1, help='Experiment Repetition Count')
    parser.add_argument('--discount', type=float, default=0.9, help=' Discount')
    parser.add_argument('--mem_len', type=float, default=10000, help=' Size of Experience Replay Memory')
    parser.add_argument('--batch_size', type=float, default=32, help=' Batch size ')
    parser.add_argument('--episode_max_steps', type=float, default=50, help='Maximum Number of steps in an episode')
    parser.add_argument('--train', action='store_true', default=False, help=' Trains all the solvers for given env.')
    parser.add_argument('--test', action='store_true', default=False,
                        help=' Restores and Tests all the solvers for given env.')

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    args.result_dir = os.path.join(args.result_dir, args.env)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # initialize environment
    env_fn = lambda: gym.make(args.env)
    env = env_fn()
    state = env.reset()
    actions = env.action_space.n
    reward_types = len(env.reward_types)

    # initialize solvers
    dr_qlearn = DRQLearn(env_fn(), args.lr, args.discount, args.min_eps, args.max_eps, args.total_episodes)
    dr_sarsa = DRSarsa(env_fn(), args.lr, args.discount, args.min_eps, args.max_eps, args.total_episodes)

    dr_dqn_model = DRModel(state.size, actions, reward_types)
    dr_dqn_solver = DRDQN(env_fn(), dr_dqn_model, args.lr, args.discount, args.mem_len, args.batch_size, args.min_eps,
                          args.max_eps, args.total_episodes)

    dr_dsarsa_model = DRModel(state.size, actions, reward_types)
    dr_dsarsa_solver = DRDSarsa(env_fn(), dr_dsarsa_model, args.lr, args.discount, args.mem_len, args.batch_size,
                                args.min_eps, args.max_eps, args.total_episodes)

    hra_model = DRModel(state.size, actions, reward_types)
    hra_solver = HRA(env_fn(), hra_model, args.lr, args.discount, args.mem_len, args.batch_size, args.min_eps,
                     args.max_eps, args.total_episodes)
    # solvers = [dr_qlearn, dr_sarsa, dr_dqn_solver, dr_dsarsa_solver, hra_solver]
    # solvers = [dr_dqn_solver, dr_dsarsa_solver, hra_solver]
    solvers = [dr_qlearn, dr_sarsa, dr_dqn_solver, dr_dsarsa_solver, hra_solver]
    # Fire it up!
    # print(env.action_meanings)
    # print(sorted(env.reward_types))
    if args.train:
        monitor.run(env_fn(), solvers, args.runs, args.total_episodes, args.eval_episodes, args.episode_max_steps,
                    args.train_interval, result_path=args.result_dir)
    if args.test:
        monitor.eval(env_fn(), solvers, args.eval_episodes, args.episode_max_steps, render=True,
                     result_path=args.result_dir)
