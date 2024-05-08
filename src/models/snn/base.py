import torch
from torch import Tensor
from control_stork.models import RecurrentSpikingModel
from control_stork.nodes import (
    CellGroup,
    InputGroup,
    FastLIFGroup,
    FastReadoutGroup,
    DirectReadoutGroup,
    TimeAverageReadoutGroup,
)
from control_stork.connections import Connection
from control_stork.initializers import (
    Initializer,
    FluctuationDrivenCenteredNormalInitializer,
    AverageInitializer,
)
from control_stork.monitors import (
    PlotStateMonitor,
    PopulationSpikeCountMonitor,
    ActiveNeuronMonitor,
    PropertyMonitor,
)
from control_stork.plotting import plot_spikes, plot_traces
from control_stork.activations import SigmoidSpike
from control_stork.layers import Layer
from src.extratypes import *


class BaseRSNN(torch.nn.Module):
    """Base class for spiking models with a hidden state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, list[int]],
        output_dim: int,
        num_rec_layers: int = 0,
        num_ff_layers: int = 1,
        dt: float = 1e-3,
        repeat_input: int = 1,
        out_style: str = "last",
        input_type: CellGroup = InputGroup,
        input_kwargs: dict = {},
        neuron_type: CellGroup = FastLIFGroup,
        neuron_kwargs: dict = {},
        readout_type: CellGroup = FastReadoutGroup,
        readout_kwargs: dict = {},
        connection_type: Connection = Connection,
        connection_kwargs: dict = {},
        activation: torch.nn.Module = SigmoidSpike,
        initializer: Initializer = FluctuationDrivenCenteredNormalInitializer(
            nu=200, sigma_u=1.0, time_step=1e-3
        ),
        regularizers: list = [],
        w_regularizers: list = [],
        activation_steepness: float = None,
        activation_bias: float = None,
        output_scale: float = None,
        device: torch.device = torch.device("cpu"),
        name: str = "RSNN",
        # **kwargs,
    ) -> None:
        super().__init__()

        # gather layer parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dt = dt
        self.num_rec_layers = num_rec_layers
        self.num_ff_layers = num_ff_layers
        self.num_layers = num_rec_layers + num_ff_layers
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * self.num_layers
        assert len(hidden_dim) == self.num_layers
        self.hidden_dim = hidden_dim
        self.repeat_input = repeat_input
        self.input_type = input_type
        self.input_kwargs = {**input_kwargs}
        self.neuron_type = neuron_type
        self.neuron_kwargs = {**neuron_kwargs}
        self.readout_type = readout_type
        self.readout_kwargs = {**readout_kwargs}
        self.connection_type = connection_type
        self.connection_kwargs = connection_kwargs
        self.activation = activation
        self.initializer = initializer
        self.regularizers = regularizers
        self.w_regularizers = w_regularizers
        assert out_style.lower() in ["mean", "last"]
        self.out_style = out_style.lower()
        self.device = device
        self.name = name

        self.input_kwargs["store_sequences"] = ["out"]
        self.neuron_kwargs["store_sequences"] = ["out", "mem"]
        self.neuron_kwargs["activation"] = self.activation
        self.readout_kwargs["store_sequences"] = ["out", "mem"]

        # make the model
        self.model = RecurrentSpikingModel(device=device)
        input_group = prev = self.model.add_group(
            self.input_type(
                self.input_dim, name=f"{self.name} Input Group", **self.input_kwargs
            )
        )
        layers = []
        first = True
        for i in range(num_rec_layers):
            new = Layer(
                name=f"{self.name} Recurrent LIF Cell Group {i+1}",
                model=self.model,
                size=self.hidden_dim[len(layers)],
                input_group=prev,
                recurrent=True,
                regs=self.regularizers,
                w_regs=self.w_regularizers,
                connection_class=Connection if first else self.connection_type,
                recurrent_connection_class=self.connection_type,
                neuron_class=self.neuron_type,
                neuron_kwargs=self.neuron_kwargs,
                connection_kwargs=dict(bias=True) if first else self.connection_kwargs,
                recurrent_connection_kwargs=self.connection_kwargs,
            )
            first = False
            layers.append(new)
            self.model.add_monitor(
                PlotStateMonitor(
                    new.output_group,
                    "out",
                    plot_fn=plot_spikes,
                    title=f"{self.name} Recurrent LIF Cell Group {i+1}",
                )
            )
            self.model.add_monitor(
                PlotStateMonitor(
                    new.output_group,
                    "mem",
                    plot_fn=plot_traces,
                    title=f"{self.name} Recurrent LIF Cell Group {i+1}",
                )
            )
            self.model.add_monitor(
                PopulationSpikeCountMonitor(new.output_group, avg=True)
            )
            self.model.add_monitor(ActiveNeuronMonitor(new.output_group))
            self.model.add_monitor(
                PropertyMonitor(
                    new.output_group,
                    "tau_mem",
                )
            )
            self.model.add_monitor(
                PropertyMonitor(
                    new.output_group,
                    "tau_syn",
                )
            )

            prev = new.output_group
        for i in range(num_ff_layers):
            new = Layer(
                name=f"{self.name} FF LIF Cell Group {i+1}",
                model=self.model,
                size=hidden_dim[len(layers)],
                input_group=prev,
                recurrent=False,
                regs=self.regularizers,
                w_regs=self.w_regularizers,
                connection_class=Connection if first else self.connection_type,
                neuron_class=self.neuron_type,
                neuron_kwargs=self.neuron_kwargs,
                connection_kwargs=dict(bias=True) if first else self.connection_kwargs,
            )
            first = False
            layers.append(new)
            self.model.add_monitor(
                PlotStateMonitor(
                    new.output_group,
                    "out",
                    plot_fn=plot_spikes,
                    title=f"{self.name} FF LIF Cell Group {i+1}",
                )
            )
            self.model.add_monitor(
                PlotStateMonitor(
                    new.output_group,
                    "mem",
                    plot_fn=plot_traces,
                    title=f"{self.name} FF LIF Cell Group {i+1}",
                )
            )
            self.model.add_monitor(
                PopulationSpikeCountMonitor(new.output_group, avg=True)
            )
            self.model.add_monitor(ActiveNeuronMonitor(new.output_group))
            self.model.add_monitor(
                PropertyMonitor(
                    new.output_group,
                    "tau_mem",
                )
            )
            self.model.add_monitor(
                PropertyMonitor(
                    new.output_group,
                    "tau_syn",
                )
            )

            prev = new.output_group

        # make the readout
        readout_connection_kwargs = self.connection_kwargs.copy()
        readout_connection_kwargs["bias"] = False
        new = Layer(
            name=f"{self.name} Readout Pool Layer",
            model=self.model,
            size=self.output_dim * readout_kwargs["n_readouts"],
            input_group=prev,
            recurrent=False,
            regs=[],
            w_regs=self.w_regularizers,
            connection_class=self.connection_type,
            neuron_class=self.readout_type,
            neuron_kwargs=self.readout_kwargs,
            connection_kwargs=readout_connection_kwargs,
        )
        layers.append(new)
        self.model.add_monitor(
            PlotStateMonitor(
                new.output_group,
                "out",
                plot_fn=plot_traces,
                title=f"{self.name} Readout Pool Layer",
            )
        )
        prev = new.output_group

        # make the readout
        if self.out_style == "mean":
            output_group = new = self.model.add_group(
                TimeAverageReadoutGroup(
                    self.output_dim,
                    steps=self.repeat_input,
                    weight_scale=1.0,
                    name=f"{self.name} Time Average Readout Group",
                )
            )

        elif self.out_style == "last":
            output_group = new = self.model.add_group(
                DirectReadoutGroup(
                    self.output_dim,
                    weight_scale=1.0,
                    name=f"{self.name} Direct Readout Group",
                )
            )

        self.model.add_monitor(
            PlotStateMonitor(
                output_group,
                "out",
                plot_fn=plot_traces,
                title=f"{self.name} Average Readout Layer",
            )
        )

        con = self.model.add_connection(
            Connection(prev, new, bias=False, requires_grad=False)
        )
        readout_initializer = AverageInitializer()

        # configure the model
        # TODO: Rework how optimizers work!
        self.model.configure(
            input_group,
            output_group,
            time_step=self.dt,
        )

        for layer in layers:
            self.initializer.initialize(layer)
        con.init_parameters(readout_initializer)

        if activation_steepness is not None:
            self.activation_steepness = torch.nn.Parameter(torch.tensor(activation_steepness), requires_grad=True)
        else:
            self.activation_steepness = 1.0

        if activation_bias is not None:
            self.activation_bias = torch.nn.Parameter(torch.tensor(activation_bias), requires_grad=True)
        else:
            self.activation_bias = 0.0
        if output_scale is not None:
            self.output_scale = torch.nn.Parameter(torch.tensor(output_scale), requires_grad=True)
        else:
            self.output_scale = 1.0

        self.model.add_monitor(
            PropertyMonitor(
                self,
                "activation_steepness",
            )
        )

        self.model.add_monitor(
            PropertyMonitor(
                self,
                "output_scale",
            )
        )

        self.optimizer = None
        self.state_initialized = False

        self.numeric_monitors = ["PopulationSpikeCountMonitor", "ActiveNeuronMonitor", "PropertyMonitor"]
        self.plot_monitors = ["PlotStateMonitor"]

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer

    def reset_state(self) -> None:
        # I work with a flag here, because I do not know the batch size at init time
        self.state_initialized = False

    def init_state(self, batch_size: int = 1) -> None:
        self.model.reset_state(batch_size)
        self.state_initialized = True

    def get_monitor_data(self, exclude: list = []) -> dict[str, object]:
        return self.model.get_monitor_data(exclude)

    def get_reg_loss(self) -> Tensor:
        return self.model.compute_regularizer_losses()

    def get_log_dict(self) -> dict[str, object]:
        log_dict = self.model.get_monitor_data()
        for key in log_dict:
            if log_dict[key] is None:
                del log_dict[key]

        return log_dict

    def to(self, device: Union[str, torch.device]) -> None:
        """move model to device"""
        self.device = device
        self.model.to(device)
        for p in self.parameters():
            if not p.device == self.device:
                p.data = p.to(self.device)

        super().to(self.device)
        return self

    def count_parameters(self):
        return self.model.count_parameters()

    def prepare_input(self, x: Tensor) -> None:
        """prepare input shapes for forward pass"""
        # add a temporal dimension if necessary
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # control stork networks want (N, T, D)
        return x.transpose(0, 1)

    def step(self, x: Tensor, record: bool = False) -> Tensor:
        """perform a single step of the model"""
        return self.output_activation(self.model(x, record=record))

    def output_activation(self, x: Tensor) -> Tensor:
        """apply the output activation function"""
        return self.output_scale * torch.tanh(self.activation_bias + self.activation_steepness * x)

    def forward(self, *args: Tensor, record: bool = False, **kwargs) -> Tensor:
        """forward pass of the model on an input sequence"""
        x = torch.cat([self.prepare_input(i) for i in args], -1)
        N = x.shape[0]
        T = x.shape[1]

        if not self.state_initialized:
            self.init_state(N)

        x_outs = torch.empty((T, N, self.output_dim), device=self.device)
        for t in range(T):
            for _ in range(self.repeat_input):
                x_out = self.step(x[:, t : t + 1], record=record)
            x_outs[t] = x_out[:, -1]

        return x_outs

    def predict(
        self,
        *args: Tensor,
        deterministic: bool = True,
        record: bool = False,
    ) -> Tensor:
        return self(*args, record=record)

    def train_fn(*args, **kwargs):
        """To be implemented by child class."""
        raise NotImplementedError

    def criterion(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __str__(self) -> str:
        return str(self.model)
