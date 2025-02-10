from pathlib import Path
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import pretty, print
from src.envs import make_env
from src.utils import (
    get_device,
    conf_to_dict,
    set_seed,
    id_generator,
)
from src.agents import PredictiveControlAgent
from src.loggers import (
    WandBLogger,
    ConsoleLogger,
    PandasLogger,
    MediaLogger
)

OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b))

@hydra.main(version_base='1.3', config_path='src/conf', config_name='config_test')
def main(cfg : DictConfig) -> None:
    
    # print the config
    print('### run config ###')
    print(OmegaConf.to_yaml(cfg))

    # get the output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = hydra_cfg['runtime']['output_dir']
    experiment_dir = Path('outputs', cfg.experiment)
    config_path = Path(out_dir, '.hydra', 'config.yaml')
    print('output directory:', out_dir)

    # make the loggers
    pretty.install()
    loggers = []
    
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
        run_id = run.id
        Path(wandb.run.dir, "hydra-config.yaml").symlink_to(config_path)
        loggers.append(WandBLogger(run))
    else:
        run = None
        run_id = id_generator()
        print('wandb logger is not used!')

    # make a console logger
    if cfg.logging.console.use:
        print('console logger is used!')
        loggers.append(ConsoleLogger())

    # make a pandas logger
    if cfg.logging.pandas.use:
        print('pandas logger is used!')
        loggers.append(PandasLogger(out_dir))

    # make an experiment pandas logger
    if cfg.logging.pandas.use:
        print('experiment pandas logger is used!')
        loggers.append(PandasLogger(experiment_dir, cfg=cfg))

    # make a media logger
    if cfg.logging.media.use:
        print('media logger is used!')
        loggers.append(MediaLogger(out_dir, run=run))
    
    # set the device
    device = get_device(cfg.device)

    # set the seed
    set_seed(cfg.seed)

    # make the environment
    env = make_env(cfg.task, cfg.seed)
    if cfg.task.num_eval_envs > 0:
        eval_env = make_env(cfg.task, eval=True, seed=1)
    else:
        eval_env = None

    # make the models
    agent = PredictiveControlAgent(
        env=env, 
        config=cfg, 
        device=device, 
        loggers=loggers,
        dir=out_dir,
        eval_env=eval_env,
        id=run_id,
    )

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

    # finish the run
    env.close()
    if cfg.logging.wandb.use:
        wandb.finish()


if __name__ == '__main__':

    main()