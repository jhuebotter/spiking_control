import torch
from control_stork import activations
from . import (
    TransitionNetRSNN,
    TransitionNetPRNN,
    PolicyNetRSNN,
    PolicyNetPRNN
)

def make_transition_model(
        action_dim: int, 
        state_dim: int, 
        config: dict, 
        verbose: bool = True
    ) -> torch.nn.Module:
    """create a policy network according to the parameters specified by the config file and task.
    
    
    """

    type_ = config['type'].lower()
    params = dict(**config['params'])
    if type_ == 'prnn':
        model = TransitionNetPRNN
    elif type_ == 'rsnn':
        model = TransitionNetRSNN
    else:
        raise NotImplementedError(f"the transition model {type_} is not implemented")

    params['activation_fn'] = make_act_fn(params.get('activation_fn', {}))

    transitionnet = model(
        action_dim=action_dim,
        state_dim=state_dim,
        **params,
    )
    if verbose: print(transitionnet)

    return transitionnet


def make_policy_model(
        action_dim: int, 
        state_dim: int,
        target_dim: int, 
        config: dict, 
        verbose: bool = True
    ) -> torch.nn.Module:
    """create a policy network according to the parameters specified by the config file and task"""

    type_ = config['type'].lower()
    params = dict(**config['params'])
    if type_ == 'prnn':
        model = PolicyNetPRNN
    elif type_ == 'rsnn':
        model = PolicyNetRSNN
    else:
        raise NotImplementedError(f"the policy model {type_} is not implemented")

    params['activation_fn'] = make_act_fn(params.get('activation_fn', {}))

    policynet = model(
        action_dim=action_dim,
        state_dim=state_dim,
        target_dim=target_dim,
        **params
    )
    if verbose: print(str(policynet))

    return policynet


def make_act_fn(params: dict) -> dict:
    af = params.get('type').lower()
    af_kwargs = params.get('activation_kwargs', {})
    if af == 'relu':
        params['act_fn'] = torch.nn.ReLU(**af_kwargs)
    elif af == 'lrelu':
        params['act_fn'] = torch.nn.LeakyReLU(**af_kwargs)
    elif af == 'sigmoid':
        params['act_fn'] = torch.nn.Sigmoid(**af_kwargs)
    elif af == 'tanh':
        params['act_fn'] = torch.nn.Tanh(**af_kwargs)
    elif af == 'sigmoidspike':
        fn = activations.SigmoidSpike
        fn.beta = af_kwargs.get('beta', fn.beta)
        fn.gamma = af_kwargs.get('gamma', fn.gamma)
        params['act_fn'] = fn
    elif af == 'gaussianspike':
        fn = activations.GaussianSpike
        fn.beta = af_kwargs.get('beta', fn.beta)
        fn.gamma = af_kwargs.get('gamma', fn.gamma)
        fn.scale = af_kwargs.get('scale', fn.scale)
        fn.hight = af_kwargs.get('hight', fn.hight)
        params['act_fn'] = fn
    elif af == 'superspike':
        fn = activations.SuperSpike
        fn.beta = af_kwargs.get('beta', fn.beta)
        fn.gamma = af_kwargs.get('gamma', fn.gamma)
        params['act_fn'] = fn
    elif af == 'default':
        if 'act_fn' in params:
            params.pop('act_fn')
    else:
        raise NotImplementedError(f"the activation function {params['act_fn']} is not implemented")

    return params