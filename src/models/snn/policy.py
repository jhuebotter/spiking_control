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


class PolicyNetRSNN(BaseRSNN):
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_rec_layers: int = 0,
        num_ff_layers: int = 2,
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
            name="Policy model",
            **kwargs,
        )

    def criterion(
        self,
        target: Tensor,
        y_hat: Tensor,
        loss_gain: Optional[dict] = None,
        relative_l2_weight: float = 1.0,
    ) -> Tensor:
        if loss_gain is None:
            L2_loss_term = torch.mean(torch.pow(target - y_hat[:, use], 2))
            L1_loss_term = torch.mean(torch.abs(target - y_hat[:, use]))
        else:
            use = torch.tensor(loss_gain["use"], device=self.device)
            gain = torch.tensor(loss_gain["gain"], device=self.device)

            L2_loss_term = torch.mean(torch.pow(target - y_hat[:, use], 2) * gain)
            L1_loss_term = torch.mean(torch.abs(target - y_hat[:, use]) * gain)
        loss = (
            relative_l2_weight * L2_loss_term + (1 - relative_l2_weight) * L1_loss_term
        )
        return loss

    def train_fn(
        self,
        memory: EpisodeMemory,
        prediction_model: BaseRSNN,
        loss_gain: Optional[dict] = None,
        batch_size: int = 128,
        warmup_steps: int = 5,
        unroll_steps: int = 20,
        action_reg_weight: float = 0.0,
        action_smoothness_reg_weight: float = 0.0,
        relative_l2_weight: float = 1.0,
        max_norm: Optional[float] = None,
        deterministic_prediction: bool = False,
        record: bool = False,
        excluded_monitor_keys: Optional[list[str]] = None,
    ) -> dict:

        # sample a batch of predictions
        (
            states,
            targets,
            actions,
            _,
            _,
            _,
        ) = memory.sample_batch(
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            unroll_steps=0,
            device=self.device,
        )

        # initialize the loss
        policy_loss = torch.zeros(1, device=self.device)

        # reset the model
        self.train()
        prediction_model.train()
        self.zero_grad()
        prediction_model.zero_grad()
        self.reset_state()
        prediction_model.reset_state()

        # warmup the model
        if warmup_steps:
            self(states[:warmup_steps], targets[:warmup_steps], record=record)
            prediction_model(states[:warmup_steps], actions[:warmup_steps])

        new_state_hat = states[-1]
        target = targets[-1]

        # unroll the model
        action_hats = []
        for i in range(unroll_steps):
            action_hat = self(new_state_hat, target, record=record)
            new_state_delta_hat = prediction_model(
                new_state_hat, action_hat, deterministic=deterministic_prediction
            )
            new_state_hat = new_state_hat + new_state_delta_hat
            policy_loss += self.criterion(
                target,
                new_state_hat.squeeze(0),
                loss_gain=loss_gain,
                relative_l2_weight=relative_l2_weight,
            )
            action_hats.append(action_hat)
        action_hats = torch.stack(action_hats, dim=1)

        # compute the loss
        policy_loss = policy_loss / unroll_steps
        reg_loss = self.get_reg_loss()
        action_reg_loss = action_hats.abs().mean() * action_reg_weight
        action_diff = action_hats[:, 1:] - action_hats[:, :-1]
        action_smoothness_reg_loss = (
            action_diff.abs().mean() * action_smoothness_reg_weight
        )
        loss = policy_loss + reg_loss + action_reg_loss + action_smoothness_reg_loss

        # update the model
        loss.backward()
        grad_norm = get_total_grad_norm(self.model)
        grad_norms = get_grad_norms(self.model)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        clipped_grad_norm = get_total_grad_norm(self.model)
        self.optimizer.step()

        result = {
            "loss": loss.item(),
            "policy loss": policy_loss.item(),
            "reg loss": reg_loss.item(),
            "action reg loss": action_reg_loss.item(),
            "action smoothness reg loss": action_smoothness_reg_loss.item(),
            "grad norm": grad_norm,
            "clipped grad norm": clipped_grad_norm,
        }

        result.update(grad_norms)

        if record:
            result.update(self.get_monitor_data(exclude=excluded_monitor_keys))

        return result
