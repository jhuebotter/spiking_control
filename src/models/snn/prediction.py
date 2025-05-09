import torch
from control_stork.activations import SigmoidSpike
from control_stork.nodes import (
    CellGroup,
    InputGroup,
    FastLIFGroup,
    ReadoutGroup,
)
from control_stork.initializers import (
    Initializer,
    FluctuationDrivenCenteredNormalInitializer,
)
from control_stork.connections import Connection
from . import BaseRSNN
from src.memory import EpisodeMemory
from src.utils import get_total_grad_norm, get_grad_norms
from src.extratypes import *


class PredictionNetRSNN(BaseRSNN):
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int,
        num_rec_layers: int = 1,
        num_ff_layers: int = 1,
        dt: float = 1e-3,
        repeat_input: int = 1,
        out_style: str = "last",
        input_type: CellGroup = InputGroup,
        input_kwargs: dict = {},
        neuron_type: CellGroup = FastLIFGroup,
        neuron_kwargs: dict = {},
        readout_type: CellGroup = ReadoutGroup,
        readout_kwargs: dict = {},
        connection_type: Connection = Connection,
        connection_kwargs: dict = {},
        activation: torch.nn.Module = SigmoidSpike,
        initializer: Initializer = FluctuationDrivenCenteredNormalInitializer(
            nu=200, sigma_u=1.0, time_step=1e-3
        ),
        regularizers: list = [],
        w_regularizers: list = [],
        device=None,
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
            dt=dt,
            repeat_input=repeat_input,
            out_style=out_style,
            input_type=input_type,
            input_kwargs=input_kwargs,
            neuron_type=neuron_type,
            neuron_kwargs=neuron_kwargs,
            readout_type=readout_type,
            readout_kwargs=readout_kwargs,
            connection_type=connection_type,
            connection_kwargs=connection_kwargs,
            activation=activation,
            initializer=initializer,
            regularizers=regularizers,
            w_regularizers=w_regularizers,
            device=device,
            name="Prediction model",
            **kwargs,
        )

    def criterion(
        self, y_hat: Tensor, y: Tensor, relative_l2_weight: float = 1.0
    ) -> Tensor:
        L2_loss_term = torch.mean(torch.pow(y - y_hat, 2))
        L1_loss_term = torch.mean(torch.abs(y - y_hat))
        loss = (
            relative_l2_weight * L2_loss_term + (1 - relative_l2_weight) * L1_loss_term
        )
        return loss

    def train_fn(
        self,
        memory: EpisodeMemory,
        batch_size: int = 128,
        warmup_steps: int = 5,
        unroll_steps: int = 1,
        teacher_forcing_p: float = 1.0,
        relative_l2_weight: float = 1.0,
        max_norm: Optional[float] = None,
        record: bool = False,
        excluded_monitor_keys: Optional[list[str]] = None,
    ) -> dict:

        # sample a batch of predictions
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
            prediction_loss += self.criterion(
                next_state_hat.squeeze(0), next_state, relative_l2_weight
            )

            state = next_state  # teacher forcing
            if teacher_forcing_p < 1.0 and torch.rand(1).item() > teacher_forcing_p:
                state = next_state_hat  # autoregressive

        # compute the loss
        prediction_loss = prediction_loss / unroll_steps
        reg_loss = self.get_reg_loss()
        loss = prediction_loss + reg_loss

        # update the model
        loss.backward()
        grad_norm = get_total_grad_norm(self.model)
        grad_norms = get_grad_norms(self.model)
        if max_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        clipped_grad_norm = get_total_grad_norm(self.model)
        self.optimizer.step()

        result = {
            "loss": loss.item(),
            "prediction loss": prediction_loss.item(),
            "reg loss": reg_loss.item(),
            "grad norm": grad_norm,
            "clipped grad norm": clipped_grad_norm,
        }

        result.update(grad_norms)

        if record:
            result.update(self.get_monitor_data(exclude=excluded_monitor_keys))

        return result
