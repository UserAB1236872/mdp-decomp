import argparse
import os
import gym, gym_decomp
import torch
import monitor
import torch.nn as nn
from algo import DRDSarsa, DRDQN, HRA
import numpy as np


class DRModel(nn.Module):
    def __init__(self, state_size, reward_types, actions):
        super(DRModel, self).__init__()
        self.actions = actions
        self.reward_types = reward_types
        self.state_size = state_size

        for rt in range(reward_types):
            model = nn.Sequential(nn.Linear(state_size, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, actions))
            model[-1].weight.data.fill_(0)
            model[-1].bias.data.fill_(0)
            setattr(self, 'model_{}'.format(rt), model)

    def forward(self, input, r_type):
        return getattr(self, 'model_{}'.format(r_type))(input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--env', default='DecomposedLunarLander-v2', help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--min_eps', type=float, default=0.1, help='Min Epsilon for Exploration')
    parser.add_argument('--max_eps', type=float, default=0.9, help='Max. Epsilon for Exploration')
    parser.add_argument('--total_episodes', type=int, default=10000, help='Total Number of episodes for training')
    parser.add_argument('--eval_episodes', type=int, default=5, help='Number of episodes for evaluation/interval')
    parser.add_argument('--train_interval', type=int, default=100, help='No. of Episodes per training interval')
    parser.add_argument('--runs', type=int, default=1, help='Experiment Repetition Count')
    parser.add_argument('--discount', type=float, default=0.99, help=' Discount')
    parser.add_argument('--mem_len', type=float, default=100000, help=' Size of Experience Replay Memory')
    parser.add_argument('--update_target_interval', type=float, default=100, help=' Batch size ')
    parser.add_argument('--batch_size', type=float, default=512, help=' Batch size ')
    parser.add_argument('--episode_max_steps', type=float, default=1000, help='Maximum Number of steps in an episode')
    parser.add_argument('--train', action='store_true', default=False, help=' Trains all the solvers for given env.')
    parser.add_argument('--test', action='store_true', default=False,
                        help=' Restores and Tests all the solvers for given env.')
    parser.add_argument('--eval_msx', action='store_true', default=False,
                        help=' Restores the solvers and Calculates MSX for state in given env.')
    parser.add_argument('--use_public_ip', action='store_true', default=False,
                        help='Loads Plottly on Public server rather than localhost')
    parser.add_argument('--host', default='127.0.0.1',
                        help='IP address for hosting the data (default: localhost)')
    parser.add_argument('--port', default='8051',
                        help='hosting port (default:8051)')
    parser.add_argument('--visualize_results', action='store_true', default=False,
                        help='Visualizes the results in the browser ')
    parser.add_argument('--use_planner', action='store_true', default=False,
                        help='Use planner for ground truth estimation of q-values (only used in table methods)')

    np.random.seed(1)
    torch.manual_seed(1)
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    args.env_result_dir = os.path.join(args.result_dir, args.env)
    if not os.path.exists(args.env_result_dir):
        os.makedirs(args.env_result_dir)

    # initialize environment
    env_fn = lambda: gym.make(args.env)
    env = env_fn()
    state = env.reset()
    actions = env.action_space.n
    reward_types = len(env.reward_types)

    # create models
    dr_model_fn = lambda: DRModel(state.size, reward_types, actions)  # decomposition model
    model_fn = lambda: DRModel(state.size, 1, actions)  # no usage of decomposition

    # create wrapper over algorithms
    dqn_solver_fn = lambda: DRDQN(env_fn(), model_fn(), args.lr, args.discount, args.mem_len, args.batch_size,
                                  args.min_eps, args.max_eps, args.total_episodes, use_cuda=args.cuda,
                                  max_episode_steps=args.episode_max_steps, use_decomposition=False,
                                  update_target_interval=args.update_target_interval)
    dsarsa_solver_fn = lambda: DRDSarsa(env_fn(), model_fn(), args.lr, args.discount, args.mem_len,
                                        args.batch_size, args.min_eps, args.max_eps, args.total_episodes,
                                        use_cuda=args.cuda, max_episode_steps=args.episode_max_steps,
                                        use_decomposition=False, update_target_interval=args.update_target_interval)

    dr_dqn_solver_fn = lambda: DRDQN(env_fn(), dr_model_fn(), args.lr, args.discount, args.mem_len, args.batch_size,
                                     args.min_eps, args.max_eps, args.total_episodes, use_cuda=args.cuda,
                                     max_episode_steps=args.episode_max_steps,
                                     update_target_interval=args.update_target_interval)

    dr_dsarsa_solver_fn = lambda: DRDSarsa(env_fn(), dr_model_fn(), args.lr, args.discount, args.mem_len,
                                           args.batch_size, args.min_eps, args.max_eps, args.total_episodes,
                                           use_cuda=args.cuda, max_episode_steps=args.episode_max_steps,
                                           update_target_interval=args.update_target_interval)

    unhra_solver_fn = lambda: HRA(env_fn(), model_fn(), args.lr, args.discount, args.mem_len, args.batch_size,
                                  args.min_eps, args.max_eps, args.total_episodes, use_cuda=args.cuda,
                                  max_episode_steps=args.episode_max_steps,
                                  update_target_interval=args.update_target_interval,
                                  use_decomposition=False)
    hra_solver_fn = lambda: HRA(env_fn(), dr_model_fn(), args.lr, args.discount, args.mem_len, args.batch_size,
                                args.min_eps, args.max_eps, args.total_episodes, use_cuda=args.cuda,
                                max_episode_steps=args.episode_max_steps,
                                update_target_interval=args.update_target_interval)

    # solvers_fn = [dqn_solver_fn, dr_dqn_solver_fn, unhra_solver_fn, hra_solver_fn, dsarsa_solver_fn,
    #               dr_dsarsa_solver_fn]
    # solvers_fn = [dr_dqn_solver_fn, hra_solver_fn, dr_dsarsa_solver_fn]
    solvers_fn = [dr_dqn_solver_fn]

    # Fire it up!
    if args.train:
        monitor.run(env_fn(), solvers_fn, args.runs, args.total_episodes, args.eval_episodes, args.episode_max_steps,
                    args.train_interval, result_path=args.env_result_dir, planner_fn=None)
    if args.test:
        for suffix in ['_best', '_last']:
            result = monitor.eval(env_fn(), solvers_fn, args.eval_episodes, args.episode_max_steps, render=False,
                                  result_path=args.env_result_dir, suffix=suffix)
            print(result)
    if args.eval_msx:
        for suffix in ['_best', '_last']:
            solvers = [s() for s in solvers_fn]
            monitor.eval_msx(env_fn(), solvers, args.eval_episodes, args.episode_max_steps,
                             result_path=args.env_result_dir, suffix=suffix)
    if args.visualize_results:
        monitor.visualize_results(result_path=args.result_dir, port=args.port, host=args.host)
