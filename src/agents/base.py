from src.utils import (
    save_checkpoint, 
    load_weights_from_disk,
    conf_to_dict,
)
from src.extratyping import *
from omegaconf import DictConfig
from gymnasium.vector import VectorEnv
import torch


class BaseAgent:
    def __init__(self, env: VectorEnv, config: DictConfig, device: torch.device):
        self.env = env
        self.config = config
        self.device = device

    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def save(
            self, 
            model: Module, 
            path: str = "model_checkpoint.cpt",
            optimizer: Optional[Optimizer] = None, 
            **kwargs
        ):
        """save model parameters to disk"""
        save_checkpoint(model, path, optimizer, **kwargs)
    
    def load(
            self,
            model: Module,
            path: str,
            optim: Optional[Optimizer] = None,
            device: str = 'cpu'
        ):
        """load model parameters from disk"""
        return load_weights_from_disk(model, path, optim, device)