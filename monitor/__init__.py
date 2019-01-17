import os
import json
import numpy as np
from .graphics import plot, msx_plot


def test(env, solver, eps, eps_max_steps, render=False, verbose=False):
    result = 0
    for ep in range(eps):
        ep_reward, ep_steps = 0, 0
        state = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action, q_values = solver.act(state, debug=True)
            state, reward, done, info = env.step(action)
            if verbose:
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
            done = done if ep_steps <= eps_max_steps else True
        result += ep_reward
    return result / eps


def run(env, solvers_fn, runs, max_eps, eval_eps, eps_max_steps, interval, result_path):
    """
    :param env_fn: creates an instance of the environment
    :param solvers: list of solvers
    :param max_eps: Maximum number of episodes to be used for training
    :param eval_eps: Number of episodes to be used during each evaluation.
    :param interval: Interval at which evaluation is done
    :param result_path: root directory to store results

    """

    info = {'test': {type(solver()).__name__: [[] for _ in range(runs)] for solver in solvers_fn},
            'test_run_mean': {type(solver()).__name__: [[] for _ in range(runs)] for solver in solvers_fn},
            'best_test': {type(solver()).__name__: [-float('inf') for _ in range(runs)] for solver in solvers_fn}}

    env.seed(0)
    for run in range(runs):
        # instantiate each solver for the run
        solvers = [solver() for solver in solvers_fn]

        curr_eps = 0
        while curr_eps < max_eps:

            # train each solver
            for solver in solvers:
                solver.train(episodes=interval)

            # evaluate each solver
            for solver in solvers:
                solver_name = type(solver).__name__
                perf = test(env, solver, eval_eps, eps_max_steps)
                info['test'][solver_name][run].append(perf)
                if len(info['test_run_mean'][solver_name][run]) == 0:
                    info['test_run_mean'][solver_name][run].append(perf)
                else:
                    n = len(info['test_run_mean'][solver_name][run])
                    m = info['test_run_mean'][solver_name][run][-1]
                    info['test_run_mean'][solver_name][run].append(((n * m + perf) / (n + 1)))

                if info['test'][solver_name][run][-1] >= info['best_test'][solver_name][run]:
                    solver.save(os.path.join(result_path, solver_name + '.p'))
                    info['best_test'][solver_name][run] = info['test'][solver_name][run][-1]

            print('Run {} : {}/{} ==> {}'.format(run, curr_eps, max_eps,
                                                 [(k, round(info['test_run_mean'][k][run][-1], 2)) for k in
                                                  sorted(info['test'].keys())]))
            curr_eps += interval

        _test_data, _test_run_mean = {}, {}
        for solver in solvers:
            solver_name = type(solver).__name__
            _test_data[solver_name] = np.average(info['test'][solver_name][:run + 1], axis=0)
            _test_run_mean[solver_name] = np.average(info['test_run_mean'][solver_name][:run + 1], axis=0)
        plot(_test_data, _test_run_mean, os.path.join(result_path, 'report.html'))


def eval(env, solvers, eval_eps, eps_max_steps, result_path, render=False):
    info = {'test': {type(solver).__name__: [] for solver in solvers}}
    # evaluate each solver
    for solver in solvers:
        env.seed(0)
        solver_name = type(solver).__name__
        solver.restore(os.path.join(result_path, solver_name + '.p'))
        perf = test(env, solver, eval_eps, eps_max_steps, render)
        info['test'][solver_name].append(perf)

    print([(s, np.average(info['test'][s])) for s in info['test']])
    return info


def eval_msx(env, solvers, optimal_solver, eps_max_steps, result_path,port,host):

    data = {'msx': [], 'actions': env.action_meanings, 'reward_types': sorted(env.reward_types)}
    optimal_solver_name = type(optimal_solver).__name__
    solver_names = []
    for solver in solvers + [optimal_solver]:
        solver_name = type(solver).__name__
        solver.restore(os.path.join(result_path, solver_name + '.p'))
        solver_names.append(solver_name)

    env.seed(0)
    steps, done = 0, False
    state = env.reset()
    while not done:
        action, q_values, msx = optimal_solver.act(state, debug=True)
        state_info = {'state': env.render(),
                      'solvers': {optimal_solver_name: {'msx': msx, 'q_values': q_values, 'action': action}}}
        for solver in solvers:
            solver_name = type(solver).__name__
            _action, _q_values, _msx = solver.act(state, debug=True)
            state_info['solvers'][solver_name] = {'msx': _msx, 'q_values': _q_values, 'action': _action}

        state, reward, done, _ = env.step(action)
        done = done or (steps >= eps_max_steps)
        steps += 1

        data['msx'].append(state_info)
    msx_plot(data['msx'], solver_names,sorted(env.reward_types), env.action_meanings, optimal_solver_name,port,host)
    return data
