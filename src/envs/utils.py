import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordVideo
from ..extratypes import *


def make_env(config: dict, seed: int = 0, eval: bool = False) -> gym.Env:
    """create an environment object containing the task.
    Args:
        config (dict): configuration file
        seed (int, optional): random seed. Defaults to 0.
        eval (bool, optional): whether to use the evaluation environment. Defaults to False.

    Returns:
        gym.Env: environment object    
    """

    task = config['type']
    if task == 'reacher':
        print('loading 2d reacher task')
        from src.envs import ReacherEnv
        env_fn = lambda : ReacherEnv(eval=eval, **config['params'])
    elif task == 'plane':
        print('loading 2d plane task')
        from src.envs import TwoDPlaneEnv
        env_fn = lambda: TwoDPlaneEnv(eval=eval, **config['params'])
    elif task == 'plane_simple':
        print('loading simple 2d plane task')
        from src.envs import TwoDPlaneEnvSimple
        env_fn = lambda: TwoDPlaneEnvSimple(eval=eval, **config['params'])
    elif task == 'franka':
        print('loading franka task')
        from src.envs.isaac_franka import FrankaEnv
        env_fn = lambda: FrankaEnv(eval=eval, num_envs=config['num_envs'], **config['params'])
        env = env_fn()
    else:
        raise NotImplementedError(f'the task {task} is not implemented')

    # wrap the environment
    if task.lower() in ['reacher', 'reacher_simple', 'plane', 'plane_simple']:
        env = wrap_env(env_fn, config, eval, seed=seed)

    return env


def wrap_env(env_fn: Callable, config: dict, eval: bool = False, seed: Optional[int] = None) -> gym.Env:
    """wrap the environment with the specified wrappers.
    Args:
        env (gym.Env): environment object
        config (dict): configuration file
        eval (bool, optional): whether to use the evaluation environment. Defaults to False.
        seed (Optional[int], optional): random seed. Defaults to None.

    Returns:
        gym.Env: wrapped environment object
    """

    if not eval:
        num_envs = config['num_envs']
    else:
        num_envs = config.get('num_eval_envs', 1)

    # wrap the environment with a vector environment
    env = SyncVectorEnv([env_fn] * num_envs)
    env.reset(seed=seed)

    return env