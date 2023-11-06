from src.utils import (
    save_checkpoint, 
    load_weights_from_disk,
    conf_to_dict,
    make_output_dir,
    id_generator,
)
from src.extratypes import *
from omegaconf import DictConfig
from gymnasium.vector import VectorEnv
import torch
from pathlib import Path
from src.loggers import PandasLogger


class BaseAgent:
    def __init__(
            self, 
            env: VectorEnv, 
            config: DictConfig, 
            device: torch.device, 
            loggers: list = [],
            dir: Optional[str] = None,
            eval_env: Optional[VectorEnv] = None,
            id: Optional[str] = None,
        ):
        self.env = env
        self.config = config
        self.loggers = loggers
        self.device = device
        self.dir = make_output_dir() if dir is None else dir
        self.eval_env = eval_env
        self.id = id_generator() if id is None else id

    def log(self, data: dict, step: Optional[int] = None) -> None:
        data.update({'id' : self.id,})
        for logger in self.loggers:
            logger.log(data, step=step)

    def run(self):
        raise NotImplementedError
    
    def finish_run(self):

        # manually save results in a PandasLogger if it exists
        for logger in self.loggers:
            if isinstance(logger, PandasLogger):
                logger.save_to_file()

    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def save(
            self, 
            model: Module, 
            dir: Optional[str] = None,
            file: Optional[str] = None,
            optimizer: Optional[Optimizer] = None, 
            **kwargs
        ) -> None:
        """save model parameters to disk"""

        if dir is None:
            dir = self.dir
        if file is None:
            file = 'model_checkpoint.cpt'
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        path = Path(dir, file)

        print(f'Saving model to {path}')
        
        save_checkpoint(model, path, optimizer, **kwargs)
    
    def load(
            self,
            model: Module,
            path: str,
            optim: Optional[Optimizer] = None,
            device: str = 'cpu'
        ) -> tuple[Module, Optional[Optimizer]]:
        """load model parameters from disk"""

        print(f'Loading model from {path}')

        return load_weights_from_disk(model, path, optim, device)