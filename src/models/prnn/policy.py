import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .base import BasePRNN
from src.memory import EpisodeMemory
from src.utils import get_total_grad_norm, get_grad_norms
from src.extratypes import *


class PolicyNetPRNN(BasePRNN):
    """probabilistic policy network"""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_rec_layers: int = 0,
        num_ff_layers: int = 2,
        bias: bool = True,
        activation: Callable = F.leaky_relu,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float,
        name: str = "Policy model",
        **kwargs
    ) -> None:

        super().__init__(
            input_dim=state_dim + target_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_rec_layers=num_rec_layers,
            num_ff_layers=num_ff_layers,
            bias=bias,
            activation=activation,
            device=device,
            dtype=dtype,
            name=name,
            **kwargs
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

    def get_reg_loss(
        self,
        action_mu: Tensor,
        action_logvar: Tensor,
        action_target_std: Optional[float] = None,
        scale: float = 1.0,
    ) -> Tensor:

        if action_target_std is None:
            return torch.zeros(1, device=self.device)

        # compute the regularization loss
        action_target_dist = torch.distributions.Normal(0.0, action_target_std)
        action_dist = torch.distributions.Normal(action_mu, action_logvar.exp().sqrt())

        action_reg_loss = (
            torch.distributions.kl_divergence(action_dist, action_target_dist)
            .mean()
            .to(self.device)
        )
        return action_reg_loss * scale

    def train_fn(
        self,
        memory: EpisodeMemory,
        prediction_model: BasePRNN,
        loss_gain: Optional[dict] = None,
        batch_size: int = 128,
        warmup_steps: int = 5,
        unroll_steps: int = 20,
        action_reg_weight: float = 0.0,
        action_smoothness_reg_weight: float = 0.0,
        relative_l2_weight: float = 1.0,
        max_norm: Optional[float] = None,
        deterministic_prediction: bool = False,
        action_target_std: Optional[float] = None,
        reg_scale: float = 1.0,
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
        reg_loss = torch.zeros(1, device=self.device)

        # reset the model
        self.train()
        prediction_model.train()
        self.zero_grad()
        prediction_model.zero_grad()
        self.reset_state()
        prediction_model.reset_state()

        # warmup the model
        if warmup_steps:
            self(states[:warmup_steps], targets[:warmup_steps])
            prediction_model(states[:warmup_steps], actions[:warmup_steps])

        new_state_hat = states[-1]
        target = targets[-1]

        # unroll the model
        action_hats = []
        for i in range(unroll_steps):
            action_mu, action_logvar = self(new_state_hat, target)
            action = self.reparameterize(action_mu, action_logvar)
            new_state_delta_hat = prediction_model.predict(
                new_state_hat, action, deterministic=deterministic_prediction
            )
            new_state_hat = new_state_hat + new_state_delta_hat
            # compute the loss
            policy_loss += self.criterion(
                target,
                new_state_hat.squeeze(0),
                loss_gain,
                relative_l2_weight=relative_l2_weight,
            )
            reg_loss += self.get_reg_loss(
                action_mu, action_logvar, action_target_std, reg_scale
            )
            action_hats.append(action_mu)
        action_hats = torch.stack(action_hats, dim=1)

        # compute the loss
        policy_loss = policy_loss / unroll_steps
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
            monitor_data = self.get_monitor_data(exclude=excluded_monitor_keys)
            result.update(monitor_data)

        return result
