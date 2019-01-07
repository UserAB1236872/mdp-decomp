from .graphics import plot
import os


def test(env, solver, eps, eps_max_steps):
    result = 0
    for ep in range(eps):
        ep_reward = 0
        ep_steps = 0
        state = env.reset()
        done = False
        while not done:
            action = solver.act(state)
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
            'test_run_mean': {type(solver).__name__: [] for solver in solvers}}
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

            print('{}/{} ==> {}'.format(curr_eps, max_eps,
                                        [(k, round(sum(info['test'][k]) / len(info['test'][k]), 2)) for k in
                                         info['test']]))
            curr_eps += interval
        plot(info['test'], info['test_run_mean'], os.path.join(result_path, 'report.html'))
