import os
import numpy as np
from .graphics import plot
from .graphics import visualize_results as vr
import pickle


def test(env, solver, eps, eps_max_steps, render=False, verbose=False):
    result = 0
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
            done = done if ep_steps <= eps_max_steps else True
        result += ep_reward
    return result / eps


def calculate_q_val_dev(env, source, target):
    dev_data = []
    for state in env.states:
        _dev = 0
        for a in range(env.action_space.n):
            _, source_action_info = source.act(state, debug=True)
            _, target_action_info = target.act(state, debug=True)
            source_q , target_q = source_action_info['q_values'],target_action_info['q_values']
            for rt in range(len(source_q)):
                for a in source_q[rt].keys():
                    _dev += abs(source_q[rt][a] - target_q[rt][a])
        dev_data.append(_dev)
    return round(sum(dev_data),2)


def run(env, solvers_fn, runs, max_eps, eval_eps, eps_max_steps, interval, result_path, planner=None):
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
            'best_test': {type(solver()).__name__: -float('inf') for solver in solvers_fn},
            'q_val_dev': {type(solver()).__name__: [[] for _ in range(runs)] for solver in solvers_fn}}

    # get optimal policy using the planner
    if planner is not None:
        planner.train()
        planner.save(os.path.join(result_path, type(planner).__name__ + '.p'))
        print(test(env, planner, eval_eps, eps_max_steps))

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
                if planner is not None:
                    dev = calculate_q_val_dev(env, planner, solver)
                    info['q_val_dev'][solver_name][run].append(dev)
                info['test'][solver_name][run].append(perf)
                if len(info['test_run_mean'][solver_name][run]) == 0:
                    info['test_run_mean'][solver_name][run].append(perf)
                else:
                    n = len(info['test_run_mean'][solver_name][run])
                    m = info['test_run_mean'][solver_name][run][-1]
                    info['test_run_mean'][solver_name][run].append(((n * m + perf) / (n + 1)))

                if info['test'][solver_name][run][-1] >= info['best_test'][solver_name]:
                    solver.save(os.path.join(result_path, solver_name + '.p'))
                    info['best_test'][solver_name] = info['test'][solver_name][run][-1]

            print('Run {} : {}/{} ==> {}'.format(run, curr_eps, max_eps,
                                                 [(k, round(info['test_run_mean'][k][run][-1], 2)) for k in
                                                  sorted(info['test'].keys())]))
            curr_eps += interval

        _test_data, _test_run_mean = {}, {}
        _q_val_dev_data = {}
        for solver in solvers:
            solver_name = type(solver).__name__
            _test_data[solver_name] = np.average(info['test'][solver_name][:run + 1], axis=0)
            _test_run_mean[solver_name] = np.average(info['test_run_mean'][solver_name][:run + 1], axis=0)
            _q_val_dev_data[solver_name] = np.average(info['q_val_dev'][solver_name][:run + 1], axis=0)
        _data = {'data': _test_data, 'run_mean': _test_run_mean, 'q_val_dev': _q_val_dev_data, 'runs': runs}
        pickle.dump(_data, open(os.path.join(result_path, 'train_data.p'), 'wb'))
        plot(_test_data, _test_run_mean, os.path.join(result_path, 'report.html'))


def eval(env, solvers_fn, eval_eps, eps_max_steps, result_path, render=False):
    data = {}
    # evaluate each solver
    for solver_fn in solvers_fn:
        solver = solver_fn()
        env.seed(0)
        solver_name = type(solver).__name__
        solver.restore(os.path.join(result_path, solver_name + '.p'))
        perf = test(env, solver, eval_eps, eps_max_steps, render)

        if solver_name in data:
            data[solver_name].append(perf)
        else:
            data[solver_name] = [perf]

    data = {k: np.average(v) for k, v in data.items()}

    return data


def eval_msx(env, solvers, eval_episodes, eps_max_steps, result_path):
    data = {}
    for solver in solvers:
        solver_name = type(solver).__name__
        solver.restore(os.path.join(result_path, solver_name + '.p'))

    env.seed(0)
    for base_solver in solvers:
        base_solver_name = type(base_solver).__name__
        solver_data = []
        for ep in range(eval_episodes):
            ep_reward = 0
            state = env.reset()
            steps, done = 0, False
            ep_data = []

            state_info = {'state': env.render(mode='print'), 'solvers': {}, 'terminal': done,
                          'reward': {_: 0 for _ in env.reward_types}}
            for solver in solvers:
                solver_name = type(solver).__name__
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
                state_info = {'state': env.render(mode='print'), 'solvers': {}, 'terminal': done,
                              'reward': step_info['reward_decomposition']}
                for solver in solvers:
                    solver_name = type(solver).__name__
                    _action, _action_info = solver.act(state, debug=True)
                    state_info['solvers'][solver_name] = _action_info
                    state_info['solvers'][solver_name]['action'] = _action
                ep_data.append(state_info)

            solver_data.append({'data': ep_data, 'score': ep_reward})
        data[base_solver_name] = solver_data

    _data = {'data': data, 'reward_types': sorted(env.reward_types), 'actions': env.action_meanings}
    pickle.dump(_data, open(os.path.join(result_path, 'x_data.p'), 'wb'))


def visualize_results(result_path, host, port):
    return vr(result_path, host, port)
