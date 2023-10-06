import gymnasium as gym
import torch


def get_grad_norm(model: torch.nn.Module) -> float:
    """calculates the total L2 norm of gradients for a model.
    This function was taken 1:1 from the pytoch forum:
    https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/2"""

    total_norm = 0.
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm


def get_device(device: str = 'cuda') -> torch.device:
    """get the device to run the code on.
    Returns:
        torch.device: device object
    """

    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def make_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """make an optimizer for a model.
    Args:
        model (torch.nn.Module): model to optimize
        config (dict): configuration file

    Returns:
        torch.optim.Optimizer: optimizer object    
    """

    optim = config['type'].lower()
    if optim == 'adam':
        Opt = torch.optim.Adam
    elif optim == 'sgd':
        Opt = torch.optim.SGD
    elif optim == 'smorms3':
        from control_stork.optimizers import SMORMS3
        Opt = SMORMS3
    else:
        raise NotImplementedError(f'The optimizer {optim} is not implemented')
    if isinstance(model, torch.nn.Module):
        return Opt(model.parameters(), **config['params'])
    elif isinstance(model, list):
        return Opt([l.parameters() for l in model], **config['params'])
