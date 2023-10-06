import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from ..extratyping import *


def make_env(config: dict, seed: int = 0) -> gym.Env:
    """create an environment object containing the task.
    Args:
        config (dict): configuration file
        seed (int, optional): random seed. Defaults to 0.

    Returns:
        gym.Env: environment object    
    """

    task = config['type']
    if task == 'reacher':
        print('loading 2d reacher task')
        from src.envs import ReacherEnv
        env_fn = lambda : ReacherEnv(**config['params'])
    elif task == 'reacher_simple':
        print('loading simple 2d reacher task')
        from src.envs import ReacherEnvSimple
        env_fn = lambda : ReacherEnvSimple(**config['params'])
    elif task == 'plane':
        print('loading 2d plane task')
        from src.envs import TwoDPlaneEnv
        env_fn = lambda: TwoDPlaneEnv(**config['params'])
    elif task == 'plane_simple':
        print('loading simple 2d plane task')
        from src.envs import TwoDPlaneEnvSimple
        env_fn = lambda: TwoDPlaneEnvSimple(**config['params'])
    else:
        raise NotImplementedError(f'the task {task} is not implemented')

    # wrap the environment
    env = wrap_env(env_fn, config)

    return env


def wrap_env(env_fn: Callable, config: dict, seed: Optional[int] = None) -> gym.Env:
    """wrap the environment with the specified wrappers.
    Args:
        env (gym.Env): environment object
        config (dict): configuration file

    Returns:
        gym.Env: wrapped environment object
    """

    # wrap the environment with a vector environment
    env = SyncVectorEnv([env_fn] * config['num_envs'])
    env.reset(seed=seed)

    return env