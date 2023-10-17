import torch
import torch.nn as nn
import torch.nn.functional as F
from src.extratypes import *


class BasePRNN(nn.Module):
    """Base class for probabilistic models that may or may not have recurrent layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_rec_layers: int = 0,
        num_ff_layers: int = 1,
        bias: bool = True,
        activation: Callable = F.leaky_relu,
        device: Union[str, torch.device] = "cpu",
        name: str = "PRNN",
        **kwargs,
    ) -> None:
        super().__init__()

        # set attributes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_rec_layers = num_rec_layers
        self.num_ff_layers = num_ff_layers
        self.bias = bias
        self.activation = activation
        self.name = name
        self.device = device

        # make layers
        layers = OrderedDict()
        if num_rec_layers:
            layers["gru"] = nn.GRU(input_dim, hidden_dim, num_rec_layers, bias)
        for i in range(num_ff_layers):
            dim = input_dim if (i == 0 and not num_rec_layers) else hidden_dim
            layers[f"fc_ff{i + 1}"] = nn.Linear(dim, hidden_dim, bias)
        layers["fc_mu"] = nn.Linear(hidden_dim, output_dim, bias)
        layers["fc_var"] = nn.Linear(hidden_dim, output_dim, bias)
        self.model = nn.ModuleDict(layers)

        # initialize weights
        nn.init.zeros_(self.model.fc_mu.bias)
        nn.init.zeros_(self.model.fc_var.bias)

        # initialize hidden state
        self.h = None

        self.optimizer = None

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer

    def reset_state(self) -> None:
        self.h = None

    def detach_state(self) -> None:
        self.h.detach_()

    def get_state(self) -> Tensor:
        return self.h

    def update_state(self, h: Tensor) -> None:
        self.h = h

    def get_monitor_data(self, exclude: list[str] = []) -> dict:
        # currently no monitor data
        return {}

    def get_reg_loss(self) -> Tensor:
        # currently no regularization
        return torch.zeros(1, device=self.device)

    def prepare_input(self, x: Tensor) -> None:
        """prepare input shapes for forward pass"""
        if self.num_rec_layers:
            # pytorch GRU expects input of shape (T, N, D)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
        else:
            # pytorch linear expects input of shape (N, D)
            if len(x.shape) == 3:
                x = x.squeeze(0)

        return x

    def to(self, device: Union[str, torch.device]) -> None:
        """move model to device and update device attribute"""
        self.device = device
        self.model.to(device)
        super().to(device)
        return self

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """perform reparametrization trick for a Gaussian distribution"""
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)

        return eps * std + mu

    def count_parameters(self):
        """count number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, *args, **kwargs) -> Tensor:
        """forward pass of the model"""
        x = torch.cat([self.prepare_input(i) for i in args], -1)

        for name, layer in self.model.items():
            if "gru" in name.lower():
                x, self.h = layer(x)
                x = self.activation(x)
            elif "fc_ff" in name.lower():
                x = self.activation(layer(x))

        mu = self.model["fc_mu"](x)
        logvar = self.model["fc_var"](x)

        return torch.tanh(mu), logvar

    def predict(
        self, *args, deterministic: bool = False, record: bool = False
    ) -> Tensor:
        """predict output of the model given input data"""
        mu, logvar = self(*args)

        if deterministic:
            return mu
        else:
            return self.reparameterize(mu, logvar)
