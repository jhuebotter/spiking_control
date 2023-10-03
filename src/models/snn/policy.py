import torch
from control_stork.activations import SigmoidSpike
from control_stork.nodes import FastLIFGroup
from . import BaseRSNN

from src.extratyping import *


class PolicyNetRSNN(BaseRSNN):
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_rec_layers: int = 0,
        num_ff_layers: int = 2,
        repeat_input: int = 1,
        out_style: str = "last",
        dt: float = 1e-3,
        device=None,
        dtype=None,
        flif_kwargs: dict = {},
        readout_kwargs: dict = {},
        neuron_type=FastLIFGroup,
        act_fn=SigmoidSpike,
        connection_dims: Optional[int] = None,
        nu: float = 50.0,
        **kwargs,
    ) -> None:
        
        # gather layer parameters
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.target_dim = target_dim

        super().__init__(
            input_dim=state_dim + target_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            num_rec_layers=num_rec_layers,
            num_ff_layers=num_ff_layers,
            repeat_input=repeat_input,
            out_style=out_style,
            dt=dt,
            device=device,
            dtype=dtype,
            flif_kwargs=flif_kwargs,
            readout_kwargs=readout_kwargs,
            neuron_type=neuron_type,
            act_fn=act_fn,
            connection_dims=connection_dims,
            nu=nu,
            name="PolicyNet",
            **kwargs,
        )

    def step(
        self, state: Tensor, target: Tensor, record: bool = False
    ) ->  Tensor:
        x = torch.cat((state, target), -1)
        x = self.basis(x, record=record)

        return torch.tanh(x)

    def forward(
        self, state: Tensor, target: Tensor, record: bool = False
    ) -> Tensor:
        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(target.shape) == 2:
            target.unsqueeze_(0)

        T = state.shape[0]
        N = state.shape[1]
        D = state.shape[2]

        # control stork networks want (N, T, D)
        state = state.transpose(0, 1)
        target = target.transpose(0, 1)

        if not self.state_initialized:
            self.init_state(N)

        mu_outs = torch.empty((T, N, self.action_dim), device=self.basis.device)
        for t in range(T):
            for _ in range(self.repeat_input):
                mu = self.step(
                    state[:, t].view(N, 1, D), target[:, t].view(N, 1, D), record=record
                )
            mu_outs[t] = mu[:, -1]

        return mu_outs

    def predict(
        self,
        state: Tensor,
        target: Tensor,
        deterministic: bool = True,
        record: bool = False,
    ) -> Tensor:
                
        return self(state, target, record)
