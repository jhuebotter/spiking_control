import torch
from control_stork.activations import SigmoidSpike
from control_stork.nodes import FastLIFGroup
from . import BaseRSNN
from src.memory import EpisodeMemory
from src.utils import get_grad_norm
from src.extratypes import *


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

    def criterion(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return torch.nn.functional.mse_loss(y_hat, y)

    def train_fn(
            self,
            memory: EpisodeMemory,
            batch_size: int = 128,
            warmup_steps: int = 5,
            unroll_steps: int = 1,
            autoregressive: bool = False,
            max_norm: Optional[float] = None,
            record: bool = False,
            excluded_monitor_keys: Optional[list[str]] = None,
        ) -> dict:

        # sample a batch of transitions
        (
            states,
            _,
            actions,
            _,
            _,
            next_states,
        ) = memory.sample_batch(
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            unroll_steps=unroll_steps,
            device=self.device,
        )

        # initialize the loss
        prediction_loss = torch.zeros(1, device=self.device)

        # reset the model
        self.train()
        self.zero_grad()
        self.reset_state()

        # warmup the model
        if warmup_steps:
            self(states[:warmup_steps], actions[:warmup_steps], record=record)
        state = states[warmup_steps]

        # unroll the model
        for i in range(unroll_steps):
            action = actions[warmup_steps + i]
            next_state = next_states[warmup_steps + i]
            next_state_delta_hat = self.predict(state, action, record=record)
            next_state_hat = state + next_state_delta_hat
            prediction_loss += self.criterion(next_state_hat.squeeze(0), next_state)
            if autoregressive:
                state = next_state_hat
            else:
                state = next_state

        # compute the loss
        prediction_loss = prediction_loss / unroll_steps
        reg_loss = self.get_reg_loss()
        loss = prediction_loss + reg_loss

        # update the model
        loss.backward()
        grad_norm = get_grad_norm(self.model)
        if max_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        clipped_grad_norm = get_grad_norm(self.model)
        self.optimizer.step()

        result = {
            "transition model loss": loss.item(),
            "transition model prediction loss": prediction_loss.item(),
            "transition model reg loss": reg_loss.item(),
            "transition model grad norm": grad_norm,
            "transition model clipped grad norm": clipped_grad_norm,
        }

        if record:
            result.update(self.get_monitor_data(exclude=excluded_monitor_keys))

        return result