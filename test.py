import os
from pathlib import Path
import torch
import celluloid
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import pygame
import tqdm
from rich.console import Console
from rich import pretty, print

from src.models import (
    PolicyNetRSNN,
    TransitionNetRSNN,
    PolicyNetPRNN,
    TransitionNetPRNN,
)
from src.envs import (
    ReacherEnv, 
    ReacherEnvSimple, 
    TwoDPlaneEnv, 
    TwoDPlaneEnvSimple,
    make_env,
)

from src.utils import (
    get_device,
    conf_to_dict,
    set_seed,
)

from src.agents import PredictiveControlAgent
from src.memory import EpisodeMemory

from src.loggers import (
    WandBLogger,
    ConsoleLogger,
    PandasLogger,
    MediaLogger
)


@hydra.main(version_base='1.3', config_path='src/conf', config_name='config')
def main(cfg : DictConfig) -> None:
    
    # print the config
    print('### run config ###')
    print(OmegaConf.to_yaml(cfg))

    # get the output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = hydra_cfg['runtime']['output_dir']
    config_path = Path(out_dir, '.hydra', 'config.yaml')
    print('output directory:', out_dir)

    #env = make_env(cfg.task, cfg.seed, eval=True)
    #steps = env.get_attr('max_episode_steps')[0]
    #f = env.call('render')
    #print('steps:', steps)
    #print(f)
    #exit()

    # make the loggers
    pretty.install()
    loggers = []
    if cfg.logging.console.use:
        print('console logger is used!')
        loggers.append(ConsoleLogger())

    # make a pandas logger
    if cfg.logging.pandas.use:
        print('pandas logger is used!')
        loggers.append(PandasLogger(out_dir))

    # make a media logger
    if cfg.logging.media.use:
        print('media logger is used!')
        loggers.append(MediaLogger(out_dir))

    # initialize wandb
    if cfg.logging.wandb.use:
        print('wandb logger is used!')
        config_dict = conf_to_dict(cfg)
        run = wandb.init(
            project=cfg.logging.wandb.project, 
            entity=cfg.logging.wandb.entity,
            config=config_dict,
            dir=out_dir,
            )
        # create a symlink to the config file
        Path(wandb.run.dir, "hydra-config.yaml").symlink_to(config_path)
        loggers.append(WandBLogger(run))
    else:
        print('wandb logger is not used!')
    
    # set the device
    device = get_device(cfg.device)

    # set the seed
    set_seed(cfg.seed)

    # make the environment
    env = make_env(cfg.task, cfg.seed)
    eval_env = make_env(cfg.task, eval=True, seed=0)

    # make the models
    agent = PredictiveControlAgent(
        env=env, 
        config=cfg, 
        device=device, 
        loggers=loggers,
        dir=out_dir,
        eval_env=eval_env,
    )

    print(agent.transition_model.name)
    print(agent.policy_model.name)

    # load the models
    load_dir = cfg.run.load_dir
    if load_dir is not None:
        load_dir = Path(load_dir)
        if not load_dir.exists():
            raise ValueError(f'load directory {load_dir} does not exist!')
        agent.load_models(load_dir)

    # watch the model with wandb
    if cfg.logging.wandb.use:
        for model in agent.models:
            wandb.watch(model)

    agent.run(cfg.learning.total_steps)

    #agent.save_models()

    # finish the run
    env.close()
    if cfg.logging.wandb.use:
        wandb.finish()
    exit()

    # test the environment
    observations, infos = env.reset()
    env.call('render')

    print('observations:', observations)
    print('infos:', infos)

    for i in range(200):
        actions = env.action_space.sample()
        observations, rewards, terminateds, truncateds, infos = env.step(actions)
        env.call('render')

        next_obs = observations['proprio']
        next_target = observations['target']
        
        print()
        print('actions:', actions)
        print('observations:', observations)
        print('rewards:', rewards)
        print('terminateds:', terminateds)
        print('truncateds:', truncateds)
        print('infos:', infos)

        if '_final_observation' in infos.keys():
            print('final observation:', infos['_final_observation'])
            final = infos['_final_observation']
            for i, f in enumerate(final):
                if f:
                    next_obs[i] = infos['final_observation'][i]['proprio']
                    next_target[i] = infos['final_observation'][i]['target']

        obs = observations['proprio']
        target = observations['target']

    env.close()


if __name__ == '__main__':

    main()