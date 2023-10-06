import torch
from control_stork.activations import SigmoidSpike
from control_stork.nodes import FastLIFGroup
from . import BaseRSNN
from src.memory import EpisodeMemory
from src.utils import get_grad_norm
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

    def criterion(self, y_hat: Tensor, y: Tensor, loss_gain: Optional[dict] = None) -> Tensor:
        if loss_gain is None:
            return torch.nn.functional.mse_loss(y_hat, y)
        use = torch.tensor(loss_gain['use'], device=self.device, dtype=torch.bool)
        gain = torch.tensor(loss_gain['gain'], device=self.device)
        return torch.mean(torch.pow(y_hat[:, use] - y, 2) * gain)

    def train_fn(
            self,
            memory: EpisodeMemory,
            transition_model: BaseRSNN,
            loss_gain: Optional[dict] = None,
            batch_size: int = 128,
            warmup_steps: int = 5,
            unroll_steps: int = 20,
            max_norm: Optional[float] = None,
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
        loss = torch.zeros(1, device=self.device)

        # reset the model
        self.train()
        transition_model.eval()
        self.zero_grad()
        transition_model.zero_grad()
        self.reset_state()
        transition_model.reset_state()

        # warmup the model
        if warmup_steps:
            self(states[:warmup_steps], targets[:warmup_steps], record=record)
            transition_model(states[:warmup_steps], actions[:warmup_steps])

        new_state_hat = states[-1]
        target = targets[-1]

        # unroll the model
        for i in range(unroll_steps):
            action_hat = self(new_state_hat, target, record=record)
            new_state_delta_hat = transition_model(new_state_hat, action_hat)
            new_state_hat = new_state_hat + new_state_delta_hat
            policy_loss += self.criterion(new_state_hat.squeeze(0), target, loss_gain=loss_gain)

        # compute the loss
        policy_loss = policy_loss / unroll_steps 
        reg_loss = self.get_reg_loss()
        loss = policy_loss + reg_loss

        # update the model
        loss.backward()
        grad_norm = get_grad_norm(self.model)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        clipped_grad_norm = get_grad_norm(self.model)
        self.optimizer.step()

        result = {
            "policy model loss": loss.item(),
            "policy model policy loss": policy_loss.item(),
            "policy model reg_loss": reg_loss.item(),
            "policy model grad norm": grad_norm,
            "policy model clipped grad norm": clipped_grad_norm,
        }

        if record:
            result.update(self.get_monitor_data(exclude=excluded_monitor_keys))

        return result
