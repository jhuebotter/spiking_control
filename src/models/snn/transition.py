import torch
from control_stork.activations import SigmoidSpike
from control_stork.nodes import FastLIFGroup
from . import BaseRSNN

from src.extratyping import *


class TransitionNetRSNN(BaseRSNN):
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int,
        num_rec_layers: int = 1,
        num_ff_layers: int = 1,
        repeat_input: int = 1,
        out_style: str = "last",
        dt: float = 0.001,
        device=None,
        dtype=None,
        flif_kwargs: dict = {},
        readout_kwargs: dict = {},
        neuron_type=FastLIFGroup,
        act_fn=SigmoidSpike,
        connection_dims: Optional[int] = None,
        nu: float = 50,
        **kwargs,
    ) -> None:
        
        self.action_dim = action_dim
        self.state_dim = state_dim

        super().__init__(
            input_dim=state_dim + action_dim,
            output_dim=state_dim,
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
            name="TransitionNet",
            **kwargs,
        )

    def step(self, state: Tensor, action: Tensor, record: bool = False) -> Tensor:
        x = torch.cat((state, action), -1)
        x = self.basis(x, record=record)

        return x
    
    def forward(self, state: Tensor, action: Tensor, record: bool = False) -> Tensor:
        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(action.shape) == 2:
            action.unsqueeze_(0)

        T = state.shape[0]
        N = state.shape[1]
        D_state = state.shape[2]
        D_action = action.shape[2]

        # control stork networks want (N, T, D)
        state = state.transpose(0, 1)
        action = action.transpose(0, 1)

        if not self.state_initialized:
            self.init_state(N)

        x_outs = torch.empty((T, N, self.output_dim), device=self.basis.device)
        for t in range(T):
            for _ in range(self.repeat_input):
                x = self.step(
                    state[:, t].view(N, 1, D_state), action[:, t].view(N, 1, D_action), record=record
                )
            x_outs[t] = x[:, -1]

        return x_outs
    
    def predict(
            self,
            state: Tensor,
            action: Tensor,
            deterministic: bool = True,
            record: bool = False,
    ) -> Tensor:
        
        return self(state, action, record)