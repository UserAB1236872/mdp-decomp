import os
import time
import numpy as np
from .graphics import plot
from .graphics import visualize_results as vr
import pickle

import hickle as hkl
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
# from multiprocessing import Pool
from math import log, floor


# Remove this function later on
# def evaluate_planner(env, solver):
#     actions = [[None for _ in range(5)] for x in range(4)]
#     for state in env.states:
#         i = int(np.where(state == 1)[0][0])
#         x, y = (int(i / 5), i - int(i / 5) * 5)
#         if x == 3 and y == 4:
#             print('Hel')
#         action, action_info = solver.act(state, debug=True)
#         actions[x][y] = env.action_meanings[action][0]
#     for row in actions:
#         print(row)

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def train(s):
    s, interval = s
    return s.train(episodes=interval)


def test(env, solver, eps, eps_max_steps, render=False, verbose=False):
    result = []
    for ep in range(eps):
        ep_reward, ep_steps = 0, 0
        state = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action, action_info = solver.act(state, debug=True)
            state, reward, done, info = env.step(action)
            if verbose:
                q_values = action_info['q_values']
                formated_q = np.concatenate((np.array(['*'] + env.action_meanings).reshape(1, 5),
                                             np.concatenate((np.array(sorted(env.reward_types)).reshape(4, 1),
                                                             np.round(q_values, 2)),
                                                            axis=1)))
                print('greedy action:{},\n Q-Values:{}\n Decomposed q_values: \n{}'.format(action,
                                                                                           np.round(q_values.sum(0), 2),
                                                                                           formated_q))
                print('Reward: {} \n Reward Decomposition: {}\n\n'.format(reward, info['reward_decomposition']))
            ep_reward += reward
            ep_steps += 1
            done = done or (ep_steps > eps_max_steps)
        # print('ep_steps: ', ep_steps, ' reward:', ep_reward)
        result.append(ep_reward)
    return result


def calculate_q_val_dev(env, source, target):
    dev_data = []
    for state in env.states:
        _dev = []
        for a in range(env.action_space.n):
            _, source_action_info = source.act(state, debug=True)
            _, target_action_info = target.act(state, debug=True)
            source_q, target_q = source_action_info['q_values'], target_action_info['q_values']
            if target.use_decomposition:
                for rt in range(len(source_q)):
                    for a in source_q[rt].keys():
                        _dev.append(abs(source_q[rt][a] - target_q[rt][a]))
            else:
                source_q = [sum(source_q[rt][a] for rt in range(len(source_q))) for a in source_q[0].keys()]
                _dev.append(abs(np.array(source_q) - np.array(target_q)))

        dev_data.append(round(np.average(_dev), 2))  # *100 bcz %age
    return round(np.average(dev_data), 2)


def run(env, solvers_fn, runs, max_eps, eval_eps, eps_max_steps, interval, result_path, planner_fn=None):
    """
    :param env_fn: creates an instance of the environment
    :param solvers: list of solvers
    :param max_eps: Maximum number of episodes to be used for training
    :param eval_eps: Number of episodes to be used during each evaluation.
    :param interval: Interval at which evaluation is done
    :param result_path: root directory to store results

    """

    info = {'test': {solver().__name__: [[] for _ in range(runs)] for solver in solvers_fn},
            'test_run_mean': {solver().__name__: [[] for _ in range(runs)] for solver in solvers_fn},
            'best_test': {solver().__name__: -float('inf') for solver in solvers_fn},
            'q_val_dev': {solver().__name__: [[] for _ in range(runs)] for solver in solvers_fn},
            'policy_eval': {solver().__name__: [[] for _ in range(runs)] for solver in solvers_fn},
            'exploration_eps': {solver().__name__: [[] for _ in range(runs)] for solver in solvers_fn},
            'experience_steps': {solver().__name__: [[] for _ in range(runs)] for solver in solvers_fn},
            'train_loss': {solver().__name__: [[] for _ in range(runs)] for solver in solvers_fn},
            'train_perf': {solver().__name__: [[] for _ in range(runs)] for solver in solvers_fn}
            }

    # get optimal policy using the planner
    if planner_fn is not None:
        planner = planner_fn()
        planner_path = os.path.join(result_path, type(planner).__name__ + '.p')
        planner.train(verbose=False)
        planner.save(planner_path)
        print("Q Iteration Performance: ",
              np.average(test(env, planner, eval_eps, eps_max_steps, render=False, verbose=False)))
        # print("Q Iteration Performance: ", evaluate_planner(env, planner))

    env.seed(0)
    for run in range(runs):
        # instantiate each solver for the run
        solvers = [solver() for solver in solvers_fn]

        curr_eps = 0
        while curr_eps < max_eps:

            # train each solver
            for solver in solvers:
                solver_name = solver.__name__
                train_perf, loss, exploration_eps, experience_steps = solver.train(episodes=interval)
                info['train_perf'][solver_name][run].append(train_perf)
                info['train_loss'][solver_name][run].append(loss)
                info['exploration_eps'][solver_name][run].append(exploration_eps)
                info['experience_steps'][solver_name][run].append(experience_steps)

            # evaluate each solver
            for solver in solvers:
                solver_name = solver.__name__
                perf = np.average(test(env, solver, eval_eps, eps_max_steps))
                if planner_fn is not None:
                    dev = calculate_q_val_dev(env, planner, solver)
                    info['q_val_dev'][solver_name][run].append(dev)

                    eval_planner = planner_fn()
                    eval_planner.policy_evaluation(policy=solver.act, verbose=False)
                    dev = calculate_q_val_dev(env, eval_planner, solver)
                    info['policy_eval'][solver_name][run].append(dev)

                info['test'][solver_name][run].append(perf)
                if len(info['test_run_mean'][solver_name][run]) == 0:
                    info['test_run_mean'][solver_name][run].append(perf)
                else:
                    n = len(info['test_run_mean'][solver_name][run])
                    m = info['test_run_mean'][solver_name][run][-1]
                    info['test_run_mean'][solver_name][run].append(((n * m + perf) / (n + 1)))

                if info['test'][solver_name][run][-1] >= info['best_test'][solver_name]:
                    solver.save(os.path.join(result_path, solver_name + '_best.p'))
                    info['best_test'][solver_name] = info['test'][solver_name][run][-1]

                solver.save(os.path.join(result_path, solver_name + '_last.p'))

            curr_eps += interval
            _msg = '{} Run {} : {}/{}' \
                   ' Test[Current Score: {} Best Score:{}]' \
                   ' Train [ Current Performance:{} Exploration:{} Loss:{} Experiance: {}'
            print(_msg.format(time.asctime(), run, curr_eps, max_eps,
                              [(k, round(info['test'][k][run][-1], 2)) for k in sorted(info['test'].keys())],
                              [(k, round(info['best_test'][k], 2)) for k in sorted(info['best_test'].keys())],
                              [(k, round(info['train_perf'][k][run][-1], 2)) for k in
                               sorted(info['train_perf'].keys())],
                              [(k, round(info['exploration_eps'][k][run][-1], 2)) for k in
                               sorted(info['exploration_eps'].keys())],
                              [(k, round(info['train_loss'][k][run][-1], 2)) for k in
                               sorted(info['train_loss'].keys())],
                              [(k, round(info['experience_steps'][k][run][-1], 2)) for k in
                               sorted(info['experience_steps'].keys())]
                              ))


        _train_perf_data, _exploration_data, _experience_steps_data, _train_loss_data = {}, {}, {}, {}
        _test_data, _test_run_mean = {}, {}

        _q_val_dev_data, _policy_eval_data = {}, {}

        info_path = os.path.join(result_path, 'info_train.p')
        if os.path.exists(info_path):
            _info = pickle.load(open(info_path, 'rb'))
            for k in info.keys():
                for s in _info[k]:
                    if s not in info[k]:
                        info[k][s] = _info[k][s]

        pickle.dump(info, open(info_path, 'wb'))

        for solver_name in info['test'].keys():
            _test_data[solver_name] = np.average(info['test'][solver_name][:run + 1], axis=0)
            _test_run_mean[solver_name] = np.average(info['test_run_mean'][solver_name][:run + 1], axis=0)
            _train_perf_data[solver_name] = np.average(info['train_perf'][solver_name][:run + 1], axis=0)
            _exploration_data[solver_name] = np.average(info['exploration_eps'][solver_name][:run + 1], axis=0)
            _experience_steps_data[solver_name] = np.average(info['experience_steps'][solver_name][:run + 1], axis=0)
            _train_loss_data[solver_name] = np.average(info['train_loss'][solver_name][:run + 1], axis=0)

            if planner_fn is not None:
                _q_val_dev_data[solver_name] = np.average(info['q_val_dev'][solver_name][:run + 1], axis=0)
                _policy_eval_data[solver_name] = np.average(info['policy_eval'][solver_name][:run + 1], axis=0)

        _data = {'data': _test_data, 'run_mean': _test_run_mean, 'q_val_dev': _q_val_dev_data,
                 'policy_eval': _policy_eval_data, 'runs': runs,
                 'train_perf': _train_perf_data, 'exploration': _exploration_data,
                 'experience': _experience_steps_data,
                 'train_loss': _train_loss_data}
        pickle.dump(_data, open(os.path.join(result_path, 'train_data.p'), 'wb'))
        plot(_test_data, _test_run_mean, os.path.join(result_path, 'report.html'))


def eval(env, solvers_fn, eval_eps, eps_max_steps, result_path, render=False, suffix='_best'):
    data = {}
    # evaluate each solver
    suffix += '.p'
    for solver_fn in solvers_fn:
        solver = solver_fn()
        env.seed(0)
        solver_name = solver.__name__
        solver.restore(os.path.join(result_path, solver_name + suffix))
        perf = np.average(test(env, solver, eval_eps, eps_max_steps, render))
        data[solver_name] = perf

    pickle.dump(data, open(os.path.join(result_path, 'test_data' + suffix), 'wb'))
    data = {k: np.average(v) for k, v in data.items()}

    return data


def eval_msx(env, solvers, eval_episodes, eps_max_steps, result_path, suffix='_best'):
    data = {}
    for solver in solvers:
        solver_name = solver.__name__
        solver.restore(os.path.join(result_path, solver_name + suffix + '.p'))

    suffix += '.p'
    env.seed(0)
    for base_solver in solvers:
        base_solver_name = base_solver.__name__
        solver_data = []
        for ep in range(eval_episodes):
            ep_reward = 0
            state = env.reset()
            steps, done = 0, False
            ep_data = []

            if 'rgb_array' in env.metadata['render.modes']:
                render_mode = 'rgb_array'
            else:
                render_mode = 'print'
            state_info = {'state': env.render(mode=render_mode),
                          'solvers': {}, 'terminal': done,
                          'reward': {_: 0 for _ in env.reward_types},
                          'render_mode': render_mode}
            for solver in solvers:
                solver_name = solver.__name__
                _action, _action_info = solver.act(state, debug=True)
                state_info['solvers'][solver_name] = _action_info
                state_info['solvers'][solver_name]['action'] = _action
            ep_data.append(state_info)

            while not done:
                action = state_info['solvers'][base_solver_name]['action']
                state, reward, done, step_info = env.step(action)
                done = done or (steps >= eps_max_steps)
                steps += 1
                ep_reward += reward
                state_info = {'state': env.render(mode=render_mode), 'solvers': {}, 'terminal': done,
                              'reward': step_info['reward_decomposition'],
                              'render_mode': render_mode}
                for solver in solvers:
                    solver_name = solver.__name__
                    _action, _action_info = solver.act(state, debug=True)
                    state_info['solvers'][solver_name] = _action_info
                    state_info['solvers'][solver_name]['action'] = _action
                ep_data.append(state_info)
            print(ep, len(ep_data))
            solver_data.append({'data': ep_data, 'score': human_format(ep_reward)})
        data[base_solver_name] = {'episodes': solver_data, 'reward_types': base_solver.reward_types}

    _data = {'data': data, 'reward_types': sorted(env.reward_types), 'actions': env.action_meanings}
    pickle.fast = True
    pickle.dump(_data, open(os.path.join(result_path, 'x_data' + suffix), 'wb'))
    # hkl.dump(_data, os.path.join(result_path, 'x_data' + suffix), mode='w',compression='gzip')


def visualize_results(result_path, host, port):
    return vr(result_path, host, port)
