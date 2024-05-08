import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .base import BasePRNN
from src.memory import EpisodeMemory
from src.utils import get_grad_norm
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
            act_fn: Callable = F.leaky_relu,
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float,
            name: str = "policy model",
            **kwargs
        ) -> None:

        super().__init__(
            input_dim=state_dim + target_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_rec_layers=num_rec_layers,
            num_ff_layers=num_ff_layers,
            bias=bias,
            act_fn=act_fn,
            device=device,
            dtype=dtype,
            name=name,
            **kwargs
        )

    def criterion(self, y_hat: Tensor, y: Tensor, loss_gain: Optional[dict] = None) -> Tensor:
        if loss_gain is None:
            return torch.nn.functional.mse_loss(y_hat, y)
        use = torch.tensor(loss_gain['use'], device=self.device, dtype=torch.bool)
        gain = torch.tensor(loss_gain['gain'], device=self.device)
        return torch.mean(torch.pow(y_hat[:, use] - y, 2) * gain)

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
        
        action_reg_loss = torch.distributions.kl_divergence(action_dist, action_target_dist).mean().to(self.device)
        return action_reg_loss * scale
    
    def train_fn(
            self,
            memory: EpisodeMemory,
            transition_model: BasePRNN,
            loss_gain: Optional[dict] = None,
            batch_size: int = 128,
            warmup_steps: int = 5,
            unroll_steps: int = 20,
            max_norm: Optional[float] = None,
            deterministic_transition: bool = False,
            action_target_std: Optional[float] = None,
            reg_scale: float = 1.0,
            record: bool = False,
            excluded_monitor_keys: Optional[list[str]] = None,
    ) -> dict:
        
        # sample a batch of transitions
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
        transition_model.train()
        self.zero_grad()
        transition_model.zero_grad()
        self.reset_state()
        transition_model.reset_state()

        # warmup the model
        if warmup_steps:
            self(states[:warmup_steps], targets[:warmup_steps])
            transition_model(states[:warmup_steps], actions[:warmup_steps])

        new_state_hat = states[-1]
        target = targets[-1]

        # unroll the model
        for i in range(unroll_steps):
            action_mu, action_logvar = self(new_state_hat, target)
            action = self.reparameterize(action_mu, action_logvar)
            new_state_delta_hat = transition_model.predict(new_state_hat, action, deterministic=deterministic_transition)
            new_state_hat = new_state_hat + new_state_delta_hat
            # compute the loss
            policy_loss += self.criterion(new_state_hat.squeeze(0), target, loss_gain)
            reg_loss += self.get_reg_loss(action_mu, action_logvar, action_target_std, reg_scale)

        # compute the loss
        policy_loss = policy_loss / unroll_steps
        reg_loss = reg_loss / unroll_steps
        loss = policy_loss + reg_loss

        # update the model
        loss.backward()

        grad_norm = get_grad_norm(self.model)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        clipped_grad_norm = get_grad_norm(self.model)
        self.optimizer.step()

        result = {
            "loss": loss.item(),
            "policy loss": policy_loss.item(),
            "reg_loss": reg_loss.item(),
            "grad norm": grad_norm,
            "clipped grad norm": clipped_grad_norm,
        }

        if record:
            monitor_data = self.get_monitor_data(exclude=excluded_monitor_keys)
            result.update(monitor_data)

        return result