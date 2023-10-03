import gymnasium as gym

def make_env(config: dict, seed: int = 0) -> gym.Env:
    """create an environment object containing the task"""

    task = config['type']
    if task == 'reacher':
        print('loading 2d reacher task')
        from src.envs import ReacherEnv
        env = ReacherEnv(seed, **config['params'])
    elif task == 'reacher_simple':
        print('loading simple 2d reacher task')
        from src.envs import ReacherEnvSimple
        env = ReacherEnvSimple(seed, **config['params'])
    elif task == 'plane':
        print('loading 2d plane task')
        from src.envs import TwoDPlaneEnv
        env = TwoDPlaneEnv(seed, **config['params'])
    elif task == 'plane_simple':
        print('loading simple 2d plane task')
        from src.envs import TwoDPlaneEnvSimple
        env = TwoDPlaneEnvSimple(seed, **config['params'])
    else:
        raise NotImplementedError(f'the task {task} is not implemented')

    return env
