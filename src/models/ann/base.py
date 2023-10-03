import torch
import torch.nn as nn
import torch.nn.functional as F
from src.extratyping import *


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
        act_fn: Callable = F.leaky_relu,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float,
        name: str = "PRNN",
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # set attributes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_rec_layers = num_rec_layers
        self.num_ff_layers = num_ff_layers
        self.bias = bias
        self.act_fn = act_fn
        self.name = name
        self.device = device

        # make layers
        layers = OrderedDict()
        if num_rec_layers:
            layers["gru"] = nn.GRU(input_dim, hidden_dim, num_rec_layers, bias)
        for i in range(num_ff_layers):
            dim = in_dim if (i == 0 and not num_rec_layers) else hidden_dim
            layers[f"fc_ff{i + 1}"] = nn.Linear(dim, hidden_dim, bias)
        layers["fc_mu"] = nn.Linear(hidden_dim, output_dim, bias)
        layers["fc_var"] = nn.Linear(hidden_dim, output_dim, bias)
        self.basis = nn.ModuleDict(layers)

        # initialize weights
        nn.init.zeros_(self.basis.fc_mu.bias)
        nn.init.zeros_(self.basis.fc_var.bias)

        # initialize hidden state
        self.h = None

    def reset_state(self) -> None:
        self.h = None

    def detach_state(self) -> None:
        self.h.detach_()

    def get_state(self) -> Tensor:
        return self.h

    def update_state(self, h: Tensor) -> None:
        self.h = h

    def prepare_input(self, x: Tensor) -> None:
        """prepare input shapes for forward pass"""
        if self.num_rec_layers:
            # pytorch GRU expects input of shape (T, N, D)
            if len(x.shape) == 2:
                x.unsqueeze_(0)
        else:
            # pytorch linear expects input of shape (N, D)
            if len(x.shape) == 3:
                x.squeeze_(0)

        return x

    def to(self, device: Union[str, torch.device]) -> None:
        """move model to device and update device attribute"""
        self.device = device
        self.basis.to(device)
        super().to(device)
        return self

    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
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

        for name, layer in self.basis.items():
            if "gru" in name.lower():
                x, self.h = layer(x)
                x = self.act_fn(x)
            elif "fc_ff" in name.lower():
                x = self.act_fn(layer(x))

        mu = self.basis["fc_mu"](x)
        logvar = self.basis["fc_var"](x)

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