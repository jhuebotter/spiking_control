import torch
import torch.nn as nn
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
    FluctuationDrivenNormalInitializer,
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


class BaseRSNN(nn.Module):
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
        input_encoder: Optional[nn.Module] = None,
        input_type: CellGroup = InputGroup,
        input_kwargs: dict = {},
        neuron_type: CellGroup = FastLIFGroup,
        neuron_kwargs: dict = {},
        readout_type: CellGroup = FastReadoutGroup,
        readout_kwargs: dict = {},
        connection_type: Connection = Connection,
        connection_kwargs: dict = {},
        activation: torch.nn.Module = SigmoidSpike,
        initializer: Initializer = FluctuationDrivenNormalInitializer(
            mu_u=0.0, nu=200, sigma_u=1.0, time_step=1e-3
        ),
        regularizers: list = [],
        w_regularizers: list = [],
        device: torch.device = torch.device("cpu"),
        name: str = "RSNN",
        # **kwargs,
    ) -> None:
        super().__init__()

        assert num_rec_layers >= 0, "Number of recurrent layers must be non-negative."
        assert num_ff_layers >= 0, "Number of feedforward layers must be non-negative."
        assert num_rec_layers + num_ff_layers > 0, "There must be at least one layer."

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

        # make the input encoder
        self.input_encoder = input_encoder
        if self.input_encoder is not None:
            self.input_dim = self.input_encoder.compute_output_shape(self.input_dim)

        # make the model
        self.model = RecurrentSpikingModel(device=device)
        self.input_group = prev = self.model.add_group(
            self.input_type(
                self.input_dim, 
                name=f"{self.name} Input Group", 
                **self.input_kwargs
            )
        )
        self.layers = []
        first = True
        for i in range(num_rec_layers):
            new = Layer(
                name=f"{self.name} Recurrent LIF Cell Group {i+1}",
                model=self.model,
                size=self.hidden_dim[len(self.layers)],
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
            self.layers.append(new)
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
                size=hidden_dim[len(self.layers)],
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
            self.layers.append(new)
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
        #readout_connection_kwargs = self.connection_kwargs.copy()
        readout_connection_kwargs = {"bias" : False}
        new = Layer(
            name=f"{self.name} Readout Layer",
            model=self.model,
            size=self.output_dim,
            input_group=prev,
            recurrent=False,
            w_regs=self.w_regularizers,
            #connection_class=self.connection_type, #! I do not want a bottleneck here
            neuron_class=self.readout_type,
            neuron_kwargs=self.readout_kwargs,
            connection_kwargs=readout_connection_kwargs,
        )
        self.layers.append(new)
        self.output_group = new.output_group

        self.model.add_monitor(
            PlotStateMonitor(
                self.output_group,
                "mem",
                plot_fn=plot_traces,
                title=f"{self.name} Readout Layer",
            )
        )
        self.model.add_monitor(
            PlotStateMonitor(
                new.output_group,
                "out",
                plot_fn=plot_traces,
                title=f"{self.name} Readout Layer",
            )
        )

        """
        # make the readout
        if self.out_style.lower() == "mean":
            self.output_group = new = self.model.add_group(
                TimeAverageReadoutGroup(
                    self.output_dim,
                    steps=self.repeat_input,
                    name=f"{self.name} Time Average Readout Group",
                    **output_kwargs,
                )
            )

        elif self.out_style.lower() == "last":
            self.output_group = new = self.model.add_group(
                DirectReadoutGroup(
                    self.output_dim,
                    name=f"{self.name} Direct Readout Group",
                    **output_kwargs,
                )
            )
        else:
            raise ValueError(f"Unknown output style {self.out_style}.")
        
        self.model.add_monitor(
            PlotStateMonitor(
                self.output_group,
                "mem",
                plot_fn=plot_traces,
                title=f"{self.name} Average Readout Layer",
            )
        )

        self.model.add_monitor(
            PlotStateMonitor(
                self.output_group,
                "out",
                plot_fn=plot_traces,
                title=f"{self.name} Average Readout Layer",
            )
        )

        con = self.model.add_connection(
            Connection(prev, new, bias=False, requires_grad=False)
        )
        readout_initializer = AverageInitializer()
        """

        # configure the model
        # TODO: Rework how optimizers work!
        self.model.configure(
            self.input_group,
            self.output_group,
            time_step=self.dt,
        )

        for layer in self.layers:
            self.initializer.initialize(layer)
        #con.init_parameters(readout_initializer)

        self.model.add_monitor(
            PropertyMonitor(
                self.input_group,
                "input_scale",
            )
        )

        self.model.add_monitor(
            PropertyMonitor(
                self.output_group,
                "weight_scale",
            )
        )

        self.model.add_monitor(
            PropertyMonitor(
                self.output_group,
                "output_scale",
            )
        )

        self.optimizer = None
        self.state_initialized = False

        self.numeric_monitors = [
            "PopulationSpikeCountMonitor",
            "ActiveNeuronMonitor",
            "PropertyMonitor",
        ]
        self.plot_monitors = ["PlotStateMonitor"]

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer

    def reset_state(self) -> None:
        # I work with a flag here, because I do not know the batch size at init time
        self.state_initialized = False

    def init_state(self, batch_size: int = 1) -> None:
        self.input_encoder.reset()
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
        """apply the output activation function, but tanh is already applied in DirectReadoutGroup"""
        return x

    def forward(self, *args: Tensor, record: bool = False, **kwargs) -> Tensor:
        """forward pass of the model on an input sequence"""
        x = torch.cat([self.prepare_input(i) for i in args], -1)
        N = x.shape[0]
        T = x.shape[1]

        if not self.state_initialized:
            self.init_state(N)

        x_outs = torch.empty((T, N, self.output_dim), device=self.device)
        for t in range(T):
            x_in = x[:, t : t + 1]
            if self.input_encoder is not None:
                x_in = self.input_encoder(x_in)
            for _ in range(self.repeat_input):
                x_out = self.step(x_in, record=record)
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
