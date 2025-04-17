import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .base import BasePRNN
from src.memory import EpisodeMemory
from src.utils import get_total_grad_norm, get_grad_norms
from src.extratypes import *


class PredictionNetPRNN(BasePRNN):
    """probabilistic prediction network"""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int,
        num_rec_layers: int = 1,
        num_ff_layers: int = 1,
        bias: bool = True,
        activation: Callable = F.leaky_relu,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float,
        name: str = "Prediction model",
        **kwargs
    ) -> None:

        super().__init__(
            input_dim=state_dim + action_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            num_rec_layers=num_rec_layers,
            num_ff_layers=num_ff_layers,
            bias=bias,
            activation=activation,
            device=device,
            dtype=dtype,
            name=name,
            **kwargs
        )

    def criterion(self, mu: Tensor, y: Tensor, logvar: Tensor) -> Tensor:
        return torch.nn.functional.gaussian_nll_loss(
            mu, y, logvar.exp(), reduction="mean"
        )

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

        # ! currently using gaussian_nll_loss means there is no need for relative_l2_weight

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
            # compute the prediction
            next_state_hat_delta_mu, next_state_hat_delta_logvar = self(
                state, action, record=record
            )
            next_state_hat_mu = state + next_state_hat_delta_mu
            # compute the loss
            prediction_loss += self.criterion(
                next_state_hat_mu, next_state, next_state_hat_delta_logvar
            )

            # update the state
            state = next_state  # teacher forcing
            if teacher_forcing_p < 1.0 and torch.rand(1).item() > teacher_forcing_p:
                next_state_hat = self.reparameterize(
                    next_state_hat_mu, next_state_hat_delta_logvar
                )
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
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
            monitor_data = self.get_monitor_data(exclude=excluded_monitor_keys)
            result.update(monitor_data)

        return result
