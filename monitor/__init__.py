from multiprocessing import Pool


def _test(env_fn, solver, seed):
    """ test a single episode """
    env = env_fn()
    env.seed(seed)
    ep_reward = []
    obs = env.reset()
    while not done:
        action = solver.act(obs)
        obs, reward, done, info = env.step(action)
        ep_reward.append(reward)
    return ep_reward


def test(env_fn, solver, eps, start_seed, processes=4):
    with Pool(processes) as p:
        result = p.map(lambda x: _test(env_fn, solver, start_seed + x), range(eps))
    return result


def run(env_fn, solvers, max_eps, eval_eps, interval, result_path):
    """
    :param env_fn: creates an instance of the environment
    :param solvers: list of solvers
    :param max_eps: Maximum number of episodes to be used for training
    :param eval_eps: Number of episodes to be used during each evaluation.
    :param interval: Interval at which evaluation is done
    :param result_path: root directory to store results

    """
    info = {solver.__name__: {'test': []} for solver in solvers}
    curr_eps = 0
    while curr_eps <= max_eps:
        # train each solver
        # Todo: perform parallely
        for solver in solvers:
            solver.train(episodes=interval)
            pass

        # evaluate each solver
        for solver in solvers:
            perf = test(env_fn, solver, eval_eps, start_seed=curr_eps)
            info[solver.__name__]['test'].append(perf)

        # save results
        curr_eps += interval
