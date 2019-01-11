import os
import numpy as np
from .graphics import plot


def test(env, solver, eps, eps_max_steps, render=False):
    result = 0
    for ep in range(eps):
        ep_reward = 0
        ep_steps = 0
        state = env.reset()
        done = False
        while not done:
            action, q_values = solver.act(state, debug=True)
            if render:
                env.render()
                formated_q = np.concatenate((np.array(['*'] + env.action_meanings).reshape(1, 5),
                                             np.concatenate((np.array(sorted(env.reward_types)).reshape(4, 1),
                                                             np.round(q_values, 2)),
                                                            axis=1)))
                print('greedy action:{},\n Q-Values:{}\n Decomposed q_values: \n{}\n\n'.format(action,
                                                                                           np.round(q_values.sum(0), 2),
                                                                                           formated_q))
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = done if ep_steps <= eps_max_steps else True
        result += ep_reward
    return result / eps


def run(env, solvers, runs, max_eps, eval_eps, eps_max_steps, interval, result_path):
    """
    :param env_fn: creates an instance of the environment
    :param solvers: list of solvers
    :param max_eps: Maximum number of episodes to be used for training
    :param eval_eps: Number of episodes to be used during each evaluation.
    :param interval: Interval at which evaluation is done
    :param result_path: root directory to store results

    """
    info = {'test': {type(solver).__name__: [] for solver in solvers},
            'test_run_mean': {type(solver).__name__: [] for solver in solvers},
            'best_test': {type(solver).__name__: -float('inf') for solver in solvers}}
    env.seed(0)
    for run in range(runs):
        curr_eps = 0
        while curr_eps < max_eps:

            # train each solver
            # Todo: parallel execution
            for solver in solvers:
                solver.train(episodes=interval)

            # evaluate each solver
            for solver in solvers:
                solver_name = type(solver).__name__
                perf = test(env, solver, eval_eps, eps_max_steps)
                info['test'][solver_name].append(perf)
                if len(info['test_run_mean'][solver_name]) == 0:
                    info['test_run_mean'][solver_name].append(perf)
                else:
                    n = len(info['test_run_mean'][solver_name])
                    m = info['test_run_mean'][solver_name][-1]
                    info['test_run_mean'][solver_name].append(((n * m + perf) / (n + 1)))

                if info['test'][solver_name][-1] >= info['best_test'][solver_name]:
                    solver.save(os.path.join(result_path, solver_name + '.p'))
                    info['best_test'][solver_name] = info['test'][solver_name][-1]

            print('{}/{} ==> {}'.format(curr_eps, max_eps,
                                        [(k, round(info['test_run_mean'][k][-1], 2)) for k in
                                         sorted(info['test'].keys())]))
            curr_eps += interval

        plot(info['test'], info['test_run_mean'], os.path.join(result_path, 'report.html'))


def eval(env, solvers, eval_eps, eps_max_steps, result_path, render=False):
    info = {'test': {type(solver).__name__: [] for solver in solvers}}
    env.seed(0)
    # evaluate each solver
    for solver in solvers:
        solver_name = type(solver).__name__
        solver.restore(os.path.join(result_path, solver_name + '.p'))
        perf = test(env, solver, eval_eps, eps_max_steps, render)
        info['test'][solver_name].append(perf)

    print([(s, np.average(info['test'][s])) for s in info['test']])
    return info
