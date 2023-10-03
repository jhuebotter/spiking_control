import torch
from torch import Tensor
from control_stork.models import RecurrentSpikingModel
from control_stork.nodes import (
    InputGroup,
    FastLIFGroup,
    FastReadoutGroup,
    DirectReadoutGroup,
    TimeAverageReadoutGroup,
)
from control_stork.connections import (
    Connection,
    BottleneckLinearConnection,
)
from control_stork.initializers import (
    FluctuationDrivenCenteredNormalInitializer,
    AverageInitializer,
)
from control_stork.monitors import (
    PlotStateMonitor, 
    PopulationSpikeCountMonitor,
    ActiveNeuronMonitor,
)
from control_stork.regularizers import (
    LowerBoundL2,
    UpperBoundL2,
    WeightL2Regularizer
)
from control_stork.plotting import (
    plot_spikes,
    plot_traces
)
from control_stork.activations import SigmoidSpike
from control_stork.layers import Layer
from src.extratyping import *


class BaseRSNN(torch.nn.Module):
    """Base class for spiking models with a hidden state. """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_rec_layers: int = 0,
        num_ff_layers: int = 1,
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
        name: str = "RSNN",
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # gather layer parameters
        assert repeat_input >= 1
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_rec_layers = num_rec_layers
        self.num_ff_layers = num_ff_layers
        self.repeat_input = repeat_input
        assert out_style.lower() in ["mean", "last"]
        self.out_style = out_style.lower()
        self.dt = dt
        self.device = device
        self.dtype = dtype
        self.nu = nu
        self.name = name

        readout_kwargs["n_readouts"] = readout_kwargs.get("n_pop", 1)
        readout_kwargs["store_sequences"] = ["out"]

        # handle regularization
        regs = []
        lowerBoundL2Strength = kwargs.get("lowerBoundL2Strength", 0.0)
        lowerBoundL2Threshold = kwargs.get("lowerBoundL2Threshold", 1e-3)
        upperBoundL2Strength = kwargs.get("upperBoundL2Strength", 0.0)
        upperBoundL2Threshold = kwargs.get("upperBoundL2Threshold", 0.3)
        if lowerBoundL2Strength > 0.0:
            regLB = LowerBoundL2(
                lowerBoundL2Strength, threshold=lowerBoundL2Threshold, dims=None
            )
            regs.append(regLB)
        if upperBoundL2Strength > 0.0:
            regUB = UpperBoundL2(
                upperBoundL2Strength, threshold=upperBoundL2Threshold, dims=1
            )
            regs.append(regUB)

        w_regs = []
        weightL2Strength = kwargs.get("weightL2Strength", 0.0)
        if weightL2Strength > 0.0:
            regW = WeightL2Regularizer(weightL2Strength)
            w_regs.append(regW)

        # make the initializers
        initializer = FluctuationDrivenCenteredNormalInitializer(
            sigma_u=1.0,
            nu=self.nu,
            time_step=self.dt,
        )

        neuron_kwargs = dict(
            tau_mem=flif_kwargs.get("V_tau_mean", 5e-3),
            tau_syn=flif_kwargs.get("I_tau_mean", 2e-3),
            activation=act_fn,
            reset=flif_kwargs.get("reset", "sub"),
            store_sequences=["out", "mem"],
        )

        if connection_dims in [None, 0]:
            connection_class = Connection
            connection_kwargs = dict(
                bias=True,
            )
        else:
            connection_class = BottleneckLinearConnection
            connection_kwargs = dict(
                bias=True,
                n_dims=connection_dims,
            )

        if "V_tau_mean" not in readout_kwargs:
            readout_kwargs["V_tau_mean"] = 5e-3
        if "I_tau_mean" not in readout_kwargs:
            readout_kwargs["I_tau_mean"] = 2e-3

        # make the model
        self.basis = RecurrentSpikingModel(device=device, dtype=dtype)
        input_group = prev = self.basis.add_group(
            InputGroup(self.input_dim, name=f"{self.name} Input Group")
        )
        first = True
        for i in range(num_rec_layers):
            new = Layer(
                name=f"{self.name} Recurrent LIF Cell Group {i+1}",
                model=self.basis,
                size=self.hidden_dim,
                input_group=prev,
                recurrent=True,
                regs=regs,
                w_regs=w_regs,
                connection_class=Connection if first else connection_class,
                recurrent_connection_class=connection_class,
                neuron_class=neuron_type,
                neuron_kwargs=neuron_kwargs,
                connection_kwargs=dict(bias=True) if first else connection_kwargs,
                recurrent_connection_kwargs=connection_kwargs,
            )
            first = False
            initializer.initialize(new)
            self.basis.add_monitor(
                PlotStateMonitor(
                    new.output_group,
                    "out",
                    plot_fn=plot_spikes,
                    title=f"{self.name} Recurrent LIF Cell Group {i+1}",
                )
            )
            self.basis.add_monitor(
                PlotStateMonitor(
                    new.output_group,
                    "mem",
                    plot_fn=plot_traces,
                    title=f"{self.name} Recurrent LIF Cell Group {i+1}",
                )
            )
            self.basis.add_monitor(
                PopulationSpikeCountMonitor(new.output_group, avg=True)
            )
            self.basis.add_monitor(ActiveNeuronMonitor(new.output_group))

            prev = new.output_group
        for i in range(num_ff_layers):
            new = Layer(
                name=f"{self.name} FF LIF Cell Group {i+1}",
                model=self.basis,
                size=hidden_dim,
                input_group=prev,
                recurrent=False,
                regs=regs,
                w_regs=w_regs,
                connection_class=Connection if first else connection_class,
                neuron_class=neuron_type,
                neuron_kwargs=neuron_kwargs,
                connection_kwargs=dict(bias=True) if first else connection_kwargs,
            )
            first = False
            initializer.initialize(new)
            self.basis.add_monitor(
                PlotStateMonitor(
                    new.output_group,
                    "out",
                    plot_fn=plot_spikes,
                    title=f"{self.name} FF LIF Cell Group {i+1}",
                )
            )
            self.basis.add_monitor(
                PlotStateMonitor(
                    new.output_group,
                    "mem",
                    plot_fn=plot_traces,
                    title=f"{self.name} FF LIF Cell Group {i+1}",
                )
            )
            self.basis.add_monitor(
                PopulationSpikeCountMonitor(new.output_group, avg=True)
            )
            self.basis.add_monitor(ActiveNeuronMonitor(new.output_group))

            prev = new.output_group

        # make the readout
        new = Layer(
            name=f"{self.name} Readout Pool Layer",
            model=self.basis,
            size=self.output_dim * readout_kwargs["n_readouts"],
            input_group=prev,
            recurrent=False,
            regs=[],
            w_regs=w_regs,
            connection_class=connection_class,
            neuron_class=FastReadoutGroup,
            neuron_kwargs=readout_kwargs,
            connection_kwargs=connection_kwargs,
        )
        initializer.initialize(new)
        self.basis.add_monitor(
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
            output_group = new = self.basis.add_group(
                TimeAverageReadoutGroup(
                    self.output_dim,
                    steps=self.repeat_input,
                    weight_scale=1.0,
                    name=f"{self.name} Time Average Readout Group",
                )
            )

        elif self.out_style == "last":
            output_group = new = self.basis.add_group(
                DirectReadoutGroup(
                    self.output_dim, weight_scale=1.0, name=f"{self.name} Direct Readout Group"
                )
            )

        self.basis.add_monitor(
            PlotStateMonitor(
                output_group, "out", plot_fn=plot_traces, title=f"{self.name} Average Readout Layer"
            )
        )

        con = self.basis.add_connection(
            Connection(prev, new, bias=False, requires_grad=False)
        )
        readout_initializer = AverageInitializer()
        con.init_parameters(readout_initializer)

        # configure the model
        # TODO: Rework how optimizers work!
        self.basis.configure(
            input_group,
            output_group,
            time_step=dt,
        )

        self.state_initialized = False

    def reset_state(self) -> None:
        # I work with a flag here, because I do not know the batch size at init time
        self.state_initialized = False

    def init_state(self, batch_size: int = 1) -> None:
        self.basis.reset_state(batch_size)
        self.state_initialized = True

    def get_monitor_data(self, exclude: list = []) -> dict[str, object]:
        return self.basis.get_monitor_data(exclude)

    def get_reg_loss(self) -> Tensor:
        return self.basis.compute_regularizer_losses()

    def get_log_dict(self) -> dict[str, object]:
        log_dict = self.basis.get_monitor_data()
        for key in log_dict:
            if log_dict[key] is None:
                del log_dict[key]

        return log_dict

    def to(self, device: Union[str, torch.device]) -> None:
        self.device = device
        self.basis.to(device)
        super().to(device)
        return self

    def count_parameters(self):
        return self.basis.count_parameters()

    def step(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def predict(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("This method should be implemented by subclasses.")