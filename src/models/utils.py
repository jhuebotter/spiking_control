import torch
from . import TransitionNetRSNN, TransitionNetPRNN, PolicyNetRSNN, PolicyNetPRNN
from control_stork import activations, initializers
from control_stork.nodes import (
    CellGroup,
    InputGroup,
    FastLIFGroup,
    AdaptiveLIFGroup,
    FastReadoutGroup,
    DirectReadoutGroup,
    TimeAverageReadoutGroup,
)
from control_stork.encoders import (
    EncoderStack,
    IdentityEncoder,
    RBFEncoder,
    LinearEncoder,
    Linear4DEncoder,
    DeltaEncoder,
)
from control_stork.connections import Connection, BottleneckLinearConnection
from control_stork.regularizers import (
    LowerBoundL1,
    LowerBoundL2,
    UpperBoundL1,
    UpperBoundL2,
    WeightL1Regularizer,
    WeightL2Regularizer,
)
from omegaconf import DictConfig, OmegaConf
from src.utils import conf_to_dict
from src.extratypes import *


def make_transition_model(
    action_dim: int, state_dim: int, config: dict, verbose: bool = True, **kwargs
) -> torch.nn.Module:
    """create a policy network according to the parameters specified by the config file and task."""

    # get the model class
    type_ = config["type"].lower()

    params = {
        "hidden_dim": config.params.get("hidden_dim", 512),
        "num_rec_layers": config.params.get("num_rec_layers", 0),
        "num_ff_layers": config.params.get("num_layers", 2) - config.params.get("num_rec_layers", 0),
        **kwargs,
    }

    # make the input encoder
    params["input_encoder"] = make_input_encoder(config.params.encoder)

    if type_ == "prnn":
        model = TransitionNetPRNN
        params.update(make_prnn_objects(config))
    elif type_ == "rsnn":
        model = TransitionNetRSNN
        params.update(make_snn_objects(config))
        params.update(
            {
                "dt": config.params.get("dt", 1e-3),
                "repeat_input": config.params.get("repeat_input", 1),
                "out_style": config.params.get("out_style", "last"),
            }
        )
    else:
        raise NotImplementedError(f"the transition model {type_} is not implemented")

    # make the activation function
    params["activation"] = make_act_fn(
        config.params.activation.get("type", "default"),
        config.params.activation.get("kwargs", {}),
    )
    if params["activation"] is None:
        params.pop("activation")

    transitionnet = model(
        action_dim=action_dim,
        state_dim=state_dim,
        **params,
    )
    if verbose:
        print(transitionnet)

    return transitionnet


def make_policy_model(
    action_dim: int,
    state_dim: int,
    target_dim: int,
    config: dict,
    verbose: bool = True,
    **kwargs,
) -> torch.nn.Module:
    """create a policy network according to the parameters specified by the config file and task"""

    type_ = config["type"].lower()
    params = {
        "hidden_dim": config.params.get("hidden_dim", 512),
        "num_rec_layers": config.params.get("num_rec_layers", 0),
        "num_ff_layers": config.params.get("num_layers", 2) - config.params.get("num_rec_layers", 0),
        **kwargs,
    }

    # make the input encoder
    params["input_encoder"] = make_input_encoder(config.params.encoder)

    if type_ == "prnn":
        model = PolicyNetPRNN
        params.update(make_prnn_objects(config))
    elif type_ == "rsnn":
        model = PolicyNetRSNN
        params.update(make_snn_objects(config))
        params.update(
            {
                "dt": config.params.get("dt", 1e-3),
                "repeat_input": config.params.get("repeat_input", 1),
                "out_style": config.params.get("out_style", "last"),
            }
        )
        # add an additional regularizer for the output layer if tanh activation is used
        if params["output_kwargs"].get("apply_tanh", None) == True:
            params["output_kwargs"].update(
                {
                    "regularizers": [
                        LowerBoundL2(
                            strength=1e-4, threshold=-3.0, basis="mem", dims=None
                        ),
                        UpperBoundL2(
                            strength=1e-4, threshold=3.0, basis="mem", dims=None
                        ),
                    ],
                    "store_sequences": ["out", "mem"],
                }
            )
    else:
        raise NotImplementedError(f"the policy model {type_} is not implemented")

    # make the activation function
    params["activation"] = make_act_fn(
        config.params.activation.get("type", "default"),
        config.params.activation.get("kwargs", {}),
    )
    if params["activation"] is None:
        params.pop("activation")

    policynet = model(
        action_dim=action_dim, state_dim=state_dim, target_dim=target_dim, **params
    )
    if verbose:
        print(str(policynet))

    return policynet


def make_prnn_objects(config: DictConfig) -> dict:

    # pretty print the configuration
    print(OmegaConf.to_yaml(config))

    object_dict = {}

    input_scaler = InputScaler(
        scaling=config.params.input.kwargs.get("scaling", 1.0),
        learn_scaling=config.params.input.kwargs.get("learn_scaling", False),
    )
    object_dict["input_scaler"] = input_scaler

    output_scaler = OutputScaler(
        weight_scale=config.params.output.kwargs.get("weight_scale", 1.0),
        output_scale=config.params.output.kwargs.get("output_scale", 1.0),
        learn_weight_scale=config.params.output.kwargs.get("learn_weight_scale", False),
        learn_output_scale=config.params.output.kwargs.get("learn_output_scale", False),
        apply_tanh=config.params.output.kwargs.get("apply_tanh", False),
    )
    object_dict["output_scaler"] = output_scaler

    return object_dict

def make_snn_objects(config: DictConfig) -> dict:
    params = {}

    # get the layer types
    params["input_type"] = get_layer_class(config.params.input.get("type", "default"))
    if params["input_type"] is None:
        params.pop("input_type")
    params["neuron_type"] = get_layer_class(config.params.neuron.get("type", "default"))
    if params["neuron_type"] is None:
        params.pop("neuron_type")
    params["readout_type"] = get_layer_class(
        config.params.readout.get("type", "default")
    )
    if params["readout_type"] is None:
        params.pop("readout_type")
    params["input_kwargs"] = config.params.input.get("kwargs", {})
    params["neuron_kwargs"] = config.params.neuron.get("kwargs", {})
    params["readout_kwargs"] = config.params.readout.get("kwargs", {})
    params["output_kwargs"] = {**config.params.output.get("kwargs", {})}

    # make the activation function
    params["activation"] = make_act_fn(
        config.params.activation.get("type", "default"),
        config.params.activation.get("kwargs", {}),
    )
    if params["activation"] is None:
        params.pop("activation")

    # get the connection type
    params["connection_type"] = get_connection_class(
        config.params.connection.get("kwargs", {}).get("n_dims", None)
    )
    params["connection_kwargs"] = config.params.connection.get("kwargs", {})
    if params["connection_kwargs"].get("n_dims", 0) in [0, None, "None"]:
        del params["connection_kwargs"].n_dims
        if "latent_bias" in params["connection_kwargs"]:
            del params["connection_kwargs"].latent_bias

    # make the initializer
    params["initializer"] = make_initilizer(
        config.params.initializer.get("type", "default"),
        **config.params.initializer.get("kwargs", {}),
    )
    if params["initializer"] is None:
        params.pop("initializer")

    # make the regularizers
    regularizers, w_regularizers = make_regularizers(config.params.regularization)
    params["regularizers"] = regularizers
    params["w_regularizers"] = w_regularizers

    return params


def make_regularizers(config: DictConfig) -> Union[list, list]:
    """make the regularizers
    Args:
        config (DictConfig): configuration file

    Returns:
        list: regularizers
        list: weight regularizers
    """

    # make activity regularizers
    regularizers = []
    for reg in config.activity:
        type = reg.type.lower()
        if reg.kwargs.get("strength", 0.0) > 0.0:
            if type == "lowerboundl2":
                reg = LowerBoundL2(**reg.kwargs)
            elif type == "upperboundl2":
                reg = UpperBoundL2(**reg.kwargs)
            elif type == "lowerboundl1":
                reg = LowerBoundL1(**reg.kwargs)
            elif type == "upperboundl1":
                reg = UpperBoundL1(**reg.kwargs)
            else:
                raise NotImplementedError(
                    f"the regularizer {reg.type} is not implemented"
                )
            regularizers.append(reg)

    # make weight regularizers
    w_regularizers = []
    for reg in config.weights:
        type = reg.type.lower()
        if reg.kwargs.get("strength", 0.0) > 0.0:
            print("found L2 regularizer with strength", reg.kwargs.get("strength", 0.0))
            if type == "l2":
                reg = WeightL2Regularizer(**reg.kwargs)
            elif type == "l1":
                reg = WeightL1Regularizer(**reg.kwargs)
            else:
                raise NotImplementedError(
                    f"the regularizer {reg.type} is not implemented"
                )
            w_regularizers.append(reg)

    return regularizers, w_regularizers


def make_input_encoder(config: DictConfig) -> EncoderStack:
    """make the input encoder
    Args:
        config (DictConfig): configuration file

    Returns:
        torch.nn.Module: input encoder
    """

    encoder_list = []
    for e in config:
        type = e.get("type", "default")
        if type is not None:
            encoder_class = get_encoder_class(type)
            encoder_kwargs = e.get("kwargs", {})
            encoder_list.append(encoder_class(**encoder_kwargs))
    return EncoderStack(encoder_list)


def get_encoder_class(type: str = "default") -> Type[torch.nn.Module]:
    type = type.lower()
    if type == "identity":
        return IdentityEncoder
    elif type == "rbf":
        return RBFEncoder
    elif type == "linear":
        return LinearEncoder
    elif type == "4d":
        return Linear4DEncoder
    elif type == "delta":
        return DeltaEncoder
    else:
        raise NotImplementedError(f"the encoder {type} is not implemented")


def get_layer_class(type: str = "default") -> Type[torch.nn.Module]:
    type = type.lower()
    if type == "default":
        return None
    elif type == "input":
        return InputGroup
    elif type == "lif":
        return FastLIFGroup
    elif type == "alif":
        return AdaptiveLIFGroup
    elif type == "readout":
        return FastReadoutGroup
    else:
        raise NotImplementedError(f"the layer {type} is not implemented")


def get_connection_class(n_dims: Optional[int] = None) -> Connection:
    """get the connection class
    Args:
        n_dims (int): number of dimensions. Defaults to None.
    Returns:
        Connection: connection class
    """

    if n_dims in [0, None, "None"]:
        print("using a normal connection")
        return Connection
    else:
        print(f"using a bottleneck connection with {n_dims} dimensions")
        return BottleneckLinearConnection


def make_initilizer(type: str = "default", **kwargs) -> initializers.Initializer:
    """make an initializer
    Args:
        type (str, optional): initializer type. Defaults to 'default'.
        kwargs (dict, optional): initializer parameters. Defaults to {}.

    Returns:
        initializers.Initializer: initializer object class
    """

    type = type.lower()
    if type == "default":
        return None
    elif type == "constant":
        return initializers.ConstantInitializer(**kwargs)
    elif type == "normal":
        return initializers.FluctuationDrivenNormalInitializer(**kwargs)
    else:
        raise NotImplementedError(f"the initializer {type} is not implemented")


def make_act_fn(
    af: str = "default", af_kwargs: dict = {}
) -> activations.SurrogateSpike:
    """make an activation function
    Args:
        af (str, optional): activation function type. Defaults to 'default'.
        af_kwargs (dict, optional): activation function parameters. Defaults to {}.

    Returns:
        activations.SurrogateSpike: activation function object class
    """

    if af == "relu":
        fn = torch.nn.ReLU(**af_kwargs)
    elif af == "lrelu":
        fn = torch.nn.LeakyReLU(**af_kwargs)
    elif af == "sigmoid":
        fn = torch.nn.Sigmoid(**af_kwargs)
    elif af == "tanh":
        fn = torch.nn.Tanh(**af_kwargs)
    elif af == "sigmoidspike":
        fn = activations.SigmoidSpike
        fn.beta = af_kwargs.get("beta", fn.beta)
        fn.gamma = af_kwargs.get("gamma", fn.gamma)
    elif af == "gaussianspike":
        fn = activations.GaussianSpike
        fn.beta = af_kwargs.get("beta", fn.beta)
        fn.gamma = af_kwargs.get("gamma", fn.gamma)
        fn.scale = af_kwargs.get("scale", fn.scale)
        fn.hight = af_kwargs.get("hight", fn.hight)
    elif af == "superspike":
        fn = activations.SuperSpike
        fn.beta = af_kwargs.get("beta", fn.beta)
        fn.gamma = af_kwargs.get("gamma", fn.gamma)
    elif af == "default":
        fn = None
    else:
        raise NotImplementedError(f"the activation function {af} is not implemented")

    return fn


class InputScaler(torch.nn.Module):
    def __init__(self, scaling: float = 1.0, learn_scaling: bool = False):
        super().__init__()

        if learn_scaling:
            self.scaling_param = torch.nn.Parameter(torch.log(torch.ones(1) * scaling))
        else:
            # here we use a buffer to store the scaling factor
            self.register_buffer("scaling", torch.ones(1) * scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.get_scaling()
    
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self.scaling_param) if hasattr(self, "scaling_param") else self.scaling
    

class OutputScaler(torch.nn.Module):
    def __init__(
        self,
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
        learn_weight_scale: bool = False,
        learn_output_scale: bool = False,
        apply_tanh: bool = False,
    ):
        super().__init__()

        if learn_weight_scale:
            self.weight_scale_param = torch.nn.Parameter(
                torch.log(torch.ones(1) * weight_scale)
            )
        else:
            self.register_buffer("weight_scale", torch.ones(1) * weight_scale)

        if learn_output_scale:
            self.output_scale_param = torch.nn.Parameter(
                torch.log(torch.ones(1) * output_scale)
            )
        else:
            self.register_buffer("output_scale", torch.ones(1) * output_scale)

        self.apply_tanh = apply_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.get_weight_scale()
        if self.apply_tanh:
            x = torch.tanh(x) * self.get_output_scale()
        return x 

    def get_weight_scale(self) -> torch.Tensor:
        return (
            torch.exp(self.weight_scale_param)
            if hasattr(self, "weight_scale_param")
            else self.weight_scale
        )

    def get_output_scale(self) -> torch.Tensor:
        return (
            torch.exp(self.output_scale_param)
            if hasattr(self, "output_scale_param")
            else self.output_scale
        )