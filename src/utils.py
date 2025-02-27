import torch
from omegaconf import DictConfig, OmegaConf
import numpy as np
from .extratypes import *
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import string
import random


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
    model: torch.nn.Module, config: dict, verbose: bool = True
) -> torch.optim.Optimizer:
    """make an optimizer for a model.
    Args:
        model (torch.nn.Module): model to optimize
        config (dict): configuration file

    Returns:
        torch.optim.Optimizer: optimizer object
    """

    # get the optimizer class
    optim = config["type"].lower()
    if optim == "adam":
        Opt = torch.optim.Adam
    elif optim == "sgd":
        Opt = torch.optim.SGD
    elif optim == "smorms3":
        from control_stork.optimizers import SMORMS3

        Opt = SMORMS3
    else:
        raise NotImplementedError(f"The optimizer {optim} is not implemented")

    # make the optimizer
    if isinstance(model, torch.nn.Module):
        o = Opt(model.parameters(), **config["params"])
    elif isinstance(model, list):
        o = Opt([l.parameters() for l in model], **config["params"])
    else:
        raise ValueError(
            "model must be a torch.nn.Module or a list of torch.nn.Modules"
        )

    if verbose:
        print(f"Using optimizer: {o}")

    return o


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
    """
    def __init__(self, start: float, end: float, warmup_steps: int = 0):
        self.start = start
        self.end = end
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def reset(self):
        """Reset the scheduler to its initial state."""
        self.current_step = 0

    def step(self):
        """Advance one step and return the new scheduled value."""
        self.current_step += 1
        return self.get_value()

    def get_value(self):
        """Return the current scheduled value. Default is the start value."""
        return self.start


class LinearScheduler(BaseScheduler):
    """
    Linearly interpolates from start to end after an optional warmup period.
    
    For t < warmup_steps, get_value() returns start.
    For t >= warmup_steps:
      Let effective_step = t - warmup_steps.
      If effective_step >= total_steps, returns end.
      Otherwise returns:
          start + (end - start) * (effective_step / total_steps)
    
    Args:
        total_steps (int): Number of steps over which to interpolate after warmup.
                           (At effective_step == total_steps, the scheduler returns end.)
    """
    def __init__(self, start: float = 1.0, end: float = 0.0, warmup_steps: int = 0, total_steps: int = 100):
        super().__init__(start, end, warmup_steps)
        self.total_steps = total_steps

    def get_value(self):
        if self.current_step < self.warmup_steps:
            return self.start
        effective_step = self.current_step - self.warmup_steps
        if effective_step >= self.total_steps:
            return self.end
        return self.start + (self.end - self.start) * (effective_step / self.total_steps)


class ExponentialScheduler(BaseScheduler):
    """
    Exponential scheduler that supports both decay and increase, with an optional warmup period.
    
    For t < warmup_steps, get_value() returns start.
    For t >= warmup_steps:
      If start >= end (decay):
          v(t) = end + (start - end) * gamma^(t - warmup_steps)
      If start < end (increase):
          v(t) = start + (end - start) * (1 - gamma^(t - warmup_steps))
    
    This behavior is identical to your current implementation.
    
    Args:
        gamma (float): Multiplicative factor (should be between 0 and 1).
    """
    def __init__(self, start: float = 1.0, end: float = 0.0, gamma: float = 0.97, warmup_steps: int = 0):
        super().__init__(start, end, warmup_steps)
        self.gamma = gamma

    def get_value(self):
        if self.current_step < self.warmup_steps:
            return self.start
        effective_step = self.current_step - self.warmup_steps
        if self.start >= self.end:
            return self.end + (self.start - self.end) * (self.gamma ** effective_step)
        else:
            return self.start + (self.end - self.start) * (1 - self.gamma ** effective_step)


class StepScheduler(BaseScheduler):
    """
    Step scheduler that, after an optional warmup period, immediately returns a constant value.
    
    For t < warmup_steps, get_value() returns start.
    For t >= warmup_steps, get_value() returns end.
    """
    def __init__(self, start: float = 1.0, end: float = 0.0, warmup_steps: int = 0):
        super().__init__(start, end, warmup_steps)

    def get_value(self):
        if self.current_step < self.warmup_steps:
            return self.start
        return self.end