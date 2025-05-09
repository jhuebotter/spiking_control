import torch
from omegaconf import DictConfig, OmegaConf
import numpy as np
from .extratypes import *
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import string
import random
from typing import Union, Sequence, List, Any


@dataclass
class FrameStack:
    frames: list[np.ndarray] = field(default_factory=list)

    def append(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def reset(self) -> None:
        self.frames = []

    def get(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.frames[index]

    def __len__(self) -> int:
        return len(self.frames)


def id_generator(size=8, chars=string.ascii_lowercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def make_output_dir() -> Path:
    """make a unique output directory for the current run based on the current date and time"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    out_dir = Path("outputs", date_str, time_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def conf_to_dict(config: DictConfig) -> dict:
    """convert a hydra config to a dictionary
    Args:
        config (DictConfig): hydra config

    Returns:
        dict: dictionary
    """
    return OmegaConf.to_container(config, resolve=True, throw_on_missing=True)


def dict_mean(dict_list: list[dict[str, float]], prefix: str = "") -> dict:
    """for a list of dicts with the same keys and numeric values return
    a dict with the same keys and averaged values

    Args:
        dict_list (list[dict[str, number]]): list of dictionaries with the same keys and numeric values

    Returns:
        dict: dictionary with averaged values
    """

    mean_dict = {}
    if len(dict_list) > 0:
        for key, value in dict_list[0].items():
            new_key = f"{prefix}{key}"
            if isinstance(value, (float, int, np.number)):
                mean_dict[new_key] = np.mean([d[key] for d in dict_list])
            elif isinstance(value, dict):
                mean_dict[new_key] = dict_mean(
                    [d[key] for d in dict_list], prefix=f"{new_key}."
                )
            elif isinstance(value, torch.Tensor):
                mean_dict[new_key] = np.mean(
                    [d[key].numpy() for d in dict_list], axis=0
                )

    return mean_dict


def get_grad_norms(model: torch.nn.Module):
    """
    Computes the gradient norms for all parameters in a given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        dict: A dictionary with parameter names as keys and their gradient norms as values.
    """
    grad_norms = {}

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norms[f"grad norm {name}"] = torch.norm(param.grad.detach()).item()

    return grad_norms


def get_total_grad_norm(model: torch.nn.Module) -> float:
    """calculates the total L2 norm of gradients for a model.
    This function was taken 1:1 from the pytoch forum:
    https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/2

    Args:
        model (torch.nn.Module): model

    Returns:
        float: total norm of gradients
    """

    total_norm = 0.0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5

    return total_norm


def get_device(device: str = "cuda", verbose: bool = True) -> torch.device:
    """get the device to run the code on.
    Returns:
        torch.device: device object
    """

    if torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"Using device: {device}")
    return device


def set_seed(seed: int = 0) -> None:
    """(re)set the seed for torch, numpy and cudnn
    Args:
        seed (int, optional): seed. Defaults to 0.

    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def make_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
    verbose: bool = True
) -> Optimizer:
    """
    Create an optimizer for a model, supporting per-group learning rates via a single 'lr'
    entry in the config, and passing any additional optimizer kwargs equally to each group.

    Config schema:
        config["type"]  (str): "adam", "sgd", "smorms3", or "soap"
        config["params"] (dict):
            "lr": Union[float, Sequence[float]]
            ... any other optimizer keyword-arguments (e.g. weight_decay, betas, momentum)

    If 'lr' is a float, that value is used for every parameter group.
    If 'lr' is a list/sequence of floats, its length must match the number of parameter groups
    (here: weights vs taus), and each entry is applied to the corresponding group.

    Args:
        model: the torch.nn.Module whose parameters will be optimized.
        config: dictionary containing optimizer type and 'params' with "lr" and other kwargs.
        verbose: if True, prints the constructed optimizer.

    Returns:
        An Optimizer with two parameter groups:
            1) all parameters whose name does *not* contain "tau"
            2) all parameters whose name *does* contain "tau"
    """
    # select optimizer class
    opt_type = config["type"].lower()
    if opt_type == "adam":
        OptCls = torch.optim.Adam
    elif opt_type == "sgd":
        OptCls = torch.optim.SGD
    elif opt_type == "smorms3":
        from control_stork.optimizers import SMORMS3
        OptCls = SMORMS3
    elif opt_type == "soap":
        from control_stork.optimizers import SOAP
        OptCls = SOAP
    else:
        raise NotImplementedError(f"Optimizer '{opt_type}' is not implemented")

    # parse config["params"]
    params_cfg = config["params"]
    assert "lr" in params_cfg, "config['params'] must include an 'lr' entry"
    assert "tau_lr" in params_cfg, "config['params'] must include a 'tau_lr' entry"
    # extract other optimizer kwargs (excluding 'lr')
    other_kwargs = {
        k: v for k, v in params_cfg.items() if k not in ["lr", "tau_lr"]
    }

    # split model parameters
    weight_params: List[torch.nn.Parameter] = []
    tau_params:    List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "tau" in name:
            tau_params.append(p)
        elif "scale" in name:
            tau_params.append(p) # we treat scale as tau for faster training
        else:
            weight_params.append(p)

    # construct optimizer with per-group lrs and shared kwargs
    group_args = [
        {"params": weight_params, "lr": params_cfg["lr"], **other_kwargs},
        {"params": tau_params, "lr": params_cfg["tau_lr"], **other_kwargs},
    ]
    optimizer = OptCls(group_args)

    if verbose:
        print(f"Using optimizer: {optimizer}")

    return optimizer


def save_checkpoint(
    model: torch.nn.Module,
    path: str = "model_checkpoint.cpt",
    optimizer: Optional[Optimizer] = None,
    **kwargs,
) -> None:
    """save model parameters to disk

    Args:
        model (torch.nn.Module): model
        path (str, optional): path to save the model. Defaults to "model_checkpoint.cpt".
        optimizer (Optional[Optimizer], optional): optimizer. Defaults to None.

    Returns:
        None
    """

    checkpoint = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        checkpoint.update({"optimizer_state_dict": optimizer.state_dict()})

    misc = dict(**kwargs)

    checkpoint.update({"misc": misc})

    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    """load model parameters from disk

    Args:
        path (str): path to model checkpoint
        device (str, optional): device to load the model on. Defaults to 'cpu'.

    Returns:
        dict: dictionary with model parameters
    """

    return torch.load(path, map_location=torch.device(device))


def load_weights_from_disk(
    model: Module, path: str, optim: Optional[Optimizer] = None, device: str = "cpu"
) -> tuple[Module, Optional[Optimizer]]:
    """update (partial) model parameters based on previous checkpoint

    Args:
        model (Module): model
        path (str): path to checkpoint
        optim (Optional[Optimizer], optional): optimizer. Defaults to None.
        device (str, optional): device to load the model on. Defaults to 'cpu'.

    Returns:
        tuple[Module, Optional[Optimizer]]: model and optimizer (if provided)
    """

    cp = load_checkpoint(path, device)
    current_weights = model.state_dict()
    new_weights = {**current_weights}
    new_weights.update(**cp["model_state_dict"])
    model.load_state_dict(new_weights)

    if optim:
        current_state = optim.state_dict()
        current_state.update(cp["optimizer_state_dict"])
        optim.load_state_dict(current_state)

    return model, optim



class BaseScheduler:
    """
    Base scheduler class.

    Provides a default interface with reset, step, and get_value methods.
    The default get_value simply returns the start value.

    Args:
        start: float, 0-dim Tensor, or sequence for initial scheduled value(s).
        end:   float, 0-dim Tensor, or sequence for final scheduled value(s).
        warmup_steps: number of initial steps to hold at `start` before scheduling.
    """

    def __init__(
        self,
        start: Union[float, Sequence[float], Tensor],
        end:   Union[float, Sequence[float], Tensor],
        warmup_steps: int = 0,
    ):
        self.start = self._to_tensor_or_float(start)
        self.end   = self._to_tensor_or_float(end)
        self.warmup_steps = warmup_steps
        self.current_step = 0

    @staticmethod
    def _to_tensor_or_float(
        x: Union[float, Sequence[float], Tensor]
    ) -> Union[float, Tensor]:
        if isinstance(x, Tensor):
            return x
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            return torch.tensor(x)
        if isinstance(x, (int, float)):
            return x
        raise TypeError(f"Unsupported scheduler parameter type: {type(x)}")

    def reset(self) -> None:
        """Reset the scheduler to step zero."""
        self.set_step(0)

    def set_step(self, step: int) -> None:
        """Set the current step to the given non-negative integer."""
        assert step >= 0, "Step must be non-negative."
        self.current_step = step

    def step(self) -> Union[float, Tensor]:
        """Advance one step and return the new scheduled value."""
        self.current_step += 1
        return self.get_value()

    def get_value(self) -> Union[float, Tensor]:
        """Return the current scheduled value. Default: returns start."""
        return self.start


class LinearScheduler(BaseScheduler):
    """
    Linearly interpolates from start to end after an optional warmup period.

    For t < warmup_steps: returns start.
    For warmup_steps <= t < warmup_steps + decay_steps:
        start + (end - start) * ((t - warmup_steps) / decay_steps)
    For t >= warmup_steps + decay_steps: returns end.

    Args:
        start: float, 0-dim Tensor, or sequence for initial value(s).
        end:   float, 0-dim Tensor, or sequence for final value(s).
        warmup_steps: number of steps to hold at `start`.
        decay_steps:  number of steps to move from start to end after warmup.
    """

    def __init__(
        self,
        start: Union[float, Sequence[float], Tensor] = 1.0,
        end:   Union[float, Sequence[float], Tensor] = 0.0,
        warmup_steps: int = 0,
        decay_steps: int = 100,
    ): 
        super().__init__(start, end, warmup_steps)
        self.decay_steps = decay_steps

    def get_value(self) -> Union[float, Tensor]:
        if self.current_step < self.warmup_steps:
            return self.start
        t = self.current_step - self.warmup_steps
        if t >= self.decay_steps:
            return self.end
        fraction = t / self.decay_steps
        return self.start + (self.end - self.start) * fraction


class ExponentialScheduler(BaseScheduler):
    """
    Exponential scheduler that supports both decay and increase, with an optional warmup period.
    Accepts floats, zero-dim Tensors, or sequences for start, end, and gamma.

    For t < warmup_steps: returns start.
    For t >= warmup_steps:
      If start >= end (decay):
          end + (start - end) * gamma^(t - warmup_steps)
      If start < end (increase):
          start + (end - start) * (1 - gamma^(t - warmup_steps))

    Args:
        start: float, 0-dim Tensor, or sequence for initial value(s).
        end:   float, 0-dim Tensor, or sequence for final value(s).
        gamma: float, 0-dim Tensor, or sequence controlling rate (0 < gamma < 1).
        warmup_steps: number of steps to hold at `start`.
    """

    def __init__(
        self,
        start: Union[float, Sequence[float], Tensor] = 1.0,
        end:   Union[float, Sequence[float], Tensor] = 0.0,
        gamma: Union[float, Sequence[float], Tensor] = 0.97,
        warmup_steps: int = 0,
    ):
        super().__init__(start, end, warmup_steps)
        self.gamma = self._to_tensor_or_float(gamma)

    def get_value(self) -> Union[float, Tensor]:
        # before warmup: flat
        if self.current_step < self.warmup_steps:
            return self.start

        t = self.current_step - self.warmup_steps
        # scalar-only fast path
        if not torch.is_tensor(self.start) and not torch.is_tensor(self.end) and not torch.is_tensor(self.gamma):
            if self.start >= self.end:
                return self.end + (self.start - self.end) * (self.gamma ** t)
            return self.start + (self.end - self.start) * (1 - self.gamma ** t)

        # tensor path: convert floats to match tensor dtype/device
        device = None
        dtype = None
        if torch.is_tensor(self.start):
            device, dtype = self.start.device, self.start.dtype
        elif torch.is_tensor(self.end):
            device, dtype = self.end.device, self.end.dtype
        elif torch.is_tensor(self.gamma):
            device, dtype = self.gamma.device, self.gamma.dtype

        start_t = torch.as_tensor(self.start, device=device, dtype=dtype)
        end_t   = torch.as_tensor(self.end,   device=device, dtype=dtype)
        gamma_t = torch.as_tensor(self.gamma, device=device, dtype=dtype)

        v_decay    = end_t + (start_t - end_t) * (gamma_t ** t)
        v_increase = start_t + (end_t - start_t) * (1 - gamma_t ** t)
        return torch.where(start_t >= end_t, v_decay, v_increase)


class StepScheduler(BaseScheduler):
    """
    Step scheduler that, after an optional warmup period, jumps from start to end.

    For t < warmup_steps: returns start.
    For t >= warmup_steps: returns end.

    Args:
        start: float, 0-dim Tensor, or sequence for initial value(s).
        end:   float, 0-dim Tensor, or sequence for final value(s).
        warmup_steps: number of steps to hold at `start`.
    """

    def __init__(
        self,
        start: Union[float, Sequence[float], Tensor] = 1.0,
        end:   Union[float, Sequence[float], Tensor] = 0.0,
        warmup_steps: int = 0,
    ):
        super().__init__(start, end, warmup_steps)

    def get_value(self) -> Union[float, Tensor]:
        if self.current_step < self.warmup_steps:
            return self.start
        return self.end


class LRSchedulerWrapper:
    """
    A learning rate scheduler wrapper that updates the optimizer's learning rates
    using a custom scheduler.

    This wrapper takes an optimizer and a scheduler instance (a subclass of BaseScheduler).
    On each step it advances the scheduler, reads out one or more new learning rates,
    and writes them into each of the optimizer's parameter groups. It also exposes
    get_value() for logging the current learning rate(s).

    The scheduler may return a single float-like value (int, float, or 0-D Tensor),
    in which case all parameter-groups get that same LR, or an iterable of values
    (e.g., a list or 1-D Tensor) whose length must match the number of param groups.
    """

    def __init__(self, optimizer: Optimizer, scheduler: BaseScheduler) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Initialize each parameter group's LR based on scheduler's start value
        initial_lr = self.scheduler.get_value()
        self._apply_lrs(initial_lr)

    def _apply_lrs(
        self, lr_values: Union[int, float, Tensor, Sequence[Union[int, float, Tensor]]]
    ) -> Union[float, List[float]]:
        """
        Helper to assign lr_values to each param_group['lr'].
        Returns a Python float or list of floats for logging.
        """
        # Handle torch.Tensor inputs
        if torch.is_tensor(lr_values):
            if lr_values.dim() == 0:
                lr_py = lr_values.item()
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr_py
                return lr_py
            else:
                raw_list = lr_values.tolist()
        # Handle Python sequences (but not strings)
        elif isinstance(lr_values, Sequence) and not isinstance(
            lr_values, (str, bytes)
        ):
            raw_list = []
            for v in lr_values:
                if torch.is_tensor(v) and v.dim() == 0:
                    raw_list.append(v.item())
                else:
                    raw_list.append(v)
        # Handle single floats/ints
        elif isinstance(lr_values, (int, float)):
            lr_py = lr_values
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_py
            return lr_py
        else:
            raise TypeError(f"Unsupported lr_values type: {type(lr_values)}")

        # At this point raw_list is a list of Python floats (or ints)
        num_groups = len(self.optimizer.param_groups)
        if len(raw_list) != num_groups:
            raise ValueError(
                f"Got {len(raw_list)} LR values for {num_groups} parameter groups"
            )
        for pg, lr in zip(self.optimizer.param_groups, raw_list):
            pg["lr"] = lr
        return raw_list

    def step(self, epoch: int = None) -> Union[float, List[float]]:
        """
        Advance the scheduler by one step (or to a specific epoch), update
        the optimizer's learning rates, and return the new LR(s).

        Args:
            epoch (int, optional): If provided, sets the scheduler to that step
                via scheduler.set_step(epoch). Otherwise calls scheduler.step().

        Returns:
            A single float or a list of floats reflecting the new learning rates.
        """
        if epoch is not None:
            # Some schedulers support setting an absolute step
            self.scheduler.set_step(epoch)
        else:
            # Default: advance by one
            self.scheduler.step()

        new_lr = self.scheduler.get_value()
        return self._apply_lrs(new_lr)

    def get_value(self) -> Union[float, List[float]]:
        """
        Query the current learning rate(s) from the underlying scheduler,
        without modifying the optimizer.

        Returns:
            A single float or a list of floats reflecting the current LR(s).
        """
        val = self.scheduler.get_value()
        # Convert any Tensor into a Python float or list
        if torch.is_tensor(val):
            if val.dim() == 0:
                return val.item()
            return val.tolist()
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            return [v.item() if torch.is_tensor(v) and v.dim() == 0 else v for v in val]
        if isinstance(val, (int, float)):
            return val
        raise TypeError(
            f"Unsupported return type from scheduler.get_value(): {type(val)}"
        )
