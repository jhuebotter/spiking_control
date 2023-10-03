import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .base import BasePRNN

from src.extratyping import *


class TransitionNetPRNN(BasePRNN):
    """probabilistic transition network"""

    def __init__(
            self, 
            action_dim: int, 
            state_dim: int, 
            hidden_dim: int, 
            num_rec_layers: int = 1,
            num_ff_layers: int = 1, 
            bias: bool = True, 
            act_fn: Callable = F.leaky_relu,
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float,
            name: str = "PolicyNet",
            **kwargs
        ) -> None:

        super().__init__(
            input_dim=state_dim + action_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            num_rec_layers=num_rec_layers,
            num_ff_layers=num_ff_layers,
            bias=bias,
            act_fn=act_fn,
            device=device,
            dtype=dtype,
            name=name,
            **kwargs
        )