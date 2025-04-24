import torch
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from .extratypes import *
from .utils import FrameStack
from .eval_helpers import make_predictions
from .memory import Episode
from celluloid import Camera
import shutil

font_scale = 1.0

rc = {
    "lines.linewidth": 1.5,
    "figure.titleweight": "normal",
    "lines.markersize": 8,
    "figure.titlesize": 12 * font_scale,
    "axes.titlesize": 12 * font_scale,
    "axes.labelsize": 12 * font_scale,
    "font.size": 12 * font_scale,
    "axes.titlesize": 12 * font_scale,
    "axes.labelsize": 12 * font_scale,
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "legend.fontsize": 12 * font_scale,
    "legend.title_fontsize": 12 * font_scale,
    "xtick.labelsize": 12 * font_scale,
    "ytick.labelsize": 12 * font_scale,
    #"text.usetex": True,
    #"text.latex.preamble": r"\usepackage{amsmath, amssymb, cmbright}",
    "axes.unicode_minus": True,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "legend.frameon": False,
    "legend.loc": "upper center",
}

mpl.rcParams.update(rc)

def latex_available() -> bool:
    # Check for the bare minimum: a LaTeX engine and a DVI→PNG (or PS) converter
    tex = shutil.which("latex")     # or "pdflatex"
    dvipng = shutil.which("dvipng")
    # you could also check for ghostscript ("gs") if you use ps backend
    return bool(tex and dvipng)

if latex_available():
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath, amssymb, cmbright}",
    })
else:
    mpl.rcParams.update({
        "text.usetex": False,
    })


def render_video(
    framestacks: list[FrameStack],
    framerate: int = 50,
    dpi: int = 70,
    save: Optional[Union[Path, str]] = "",
    max_stacks: int = 8,
) -> object:
    """render a video from a list of numpy RGB arrays and save to disk
    Args:
        frames: list of FrameStack objects of RGB arrays
        framerate: frames per second
        dpi: dots per inch
        save: path to save the video to

    Returns:
        animation object
    """

    assert len(framestacks) > 0
    height, width, _ = framestacks[0][0].shape
    frames = []
    for i, framestack in enumerate(framestacks):
        if i >= max_stacks:
            break
        for frame in framestack:
            frames.append(frame)

    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    if save:
        print(f"Saving animation to {save}")
        anim.save(save)

    return anim


def animate_predictions(
    episodes: list[Episode],
    prediction_model: torch.nn.Module,
    labels: list[str],
    unroll: int = 1,
    warmup: int = 0,
    step: int = 10,
    deterministic: bool = True,
    fps: int = 50,
    save: Optional[Union[Path, str]] = "",
    max_animations: int = 1,
) -> object:

    predictions = make_predictions(
        prediction_model=prediction_model,
        episodes=episodes,
        warmup=warmup,
        unroll=unroll,
        step=step,
        deterministic=deterministic,
    )
    predictions = predictions.detach().cpu().numpy()
    next_states = (
        torch.stack(
            [torch.stack([s.next_state for s in episode]) for episode in episodes],
            dim=1,
        )
        .cpu()
        .numpy()
    )

    N = predictions.shape[2]

    animations = []

    for i in range(min(N, max_animations)):
        animation = animate_prediction(
            predictions[:, :, i],
            next_states[:, i],
            labels,
            warmup,
            step,
            fps,
            save,
        )
        animations.append(animation)

    return animations


def animate_prediction(
    predictions: np.ndarray,
    next_states: np.ndarray,
    labels: list[str],
    warmup: int = 0,
    step: int = 10,
    fps: int = 30,
    save: Optional[Union[Path, str]] = "",
) -> object:
    """animate predictions and save to disk
    Args:
        predictions: array of shape [T, unroll, dim]
        next_states: array of shape [T, dim]
        labels: list of labels for each dimension
        warmup: number of steps to warmup the model
        fps: frames per second
        dpi: dots per inch
        save: path to save the video to

    Returns:
        animation object
    """

    T, h, D = predictions.shape
    T = T * step

    fig, ax = plt.subplots(D, figsize=(5, D), sharex=True, sharey=True)
    plt.ylim(-1.1, 1.1)

    cmap = plt.get_cmap("plasma")

    camera = Camera(fig)

    # make an initial snapshot without prediction
    for d in range(D):
        ax[d].plot([o[d] for o in next_states], c="g", alpha=0.5)
        ax[d].set_ylabel(labels[d])
        ax[d].tick_params(axis="both", which="major")
    ax[-1].set_xlabel("step")
    plt.tight_layout()

    camera.snap()
    idx = np.arange(0.0, 1.0, 1.0 / (h))

    # animate the prediction
    for i, t in enumerate(np.arange(0, T, step)):
        for d in range(D):
            max_ = np.min([h, T - t])
            ax[d].scatter(
                np.arange(t, np.min([t + warmup, T])),
                predictions[i, : np.min([warmup, T - t]), d],
                c="k",
                s=4,
            )
            if T - t > warmup:
                ax[d].scatter(
                    np.arange(t + warmup, np.min([t + h, T])),
                    predictions[i, warmup:max_, d],
                    c=cmap(idx[: max_ - warmup]),
                    s=4,
                )
            ax[d].plot([o[d] for o in next_states], c="g", alpha=0.5)

        plt.tight_layout()
        camera.snap()

    animation = camera.animate(interval=step * 1000.0 / fps, blit=True)
    plt.close()

    if save:
        print(f"Saving animation to {save}")
        animation.save(save)  # , bitrate=-1)

    return animation


# -------------------------------
# Static Plotting Functions
# -------------------------------


def compute_distance(position, target):
    """
    Compute the Euclidean distance between corresponding points in the
    'position' and 'target' arrays over time. Works for both 2D and 3D data.

    Parameters:
      position: Array of shape (T, d), where d is 2 or 3.
      target:   Array of shape (T, d) with the same shape as 'position'.

    Returns:
      distances: Array of shape (T,), where each element is the Euclidean distance
                 between the corresponding rows in position and target.
    """
    position = np.asarray(position)
    target = np.asarray(target)

    if position.shape != target.shape:
        raise ValueError("Position and target must have the same shape.")

    distances = np.linalg.norm(position - target, axis=1)
    return distances


def make_segments(x, y):
    """
    Create an array of 2D line segments from x and y coordinates.
    Returns an array of shape (num_segments, 2, 2).
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline2d(
    x, y, z=None, cmap="rainbow", norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0
):
    """
    Plot a colored 2D line with coordinates x and y.
    Colors are taken from z (default: a linear array between 0 and 1) mapped by cmap.
    Returns the LineCollection.
    """
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )
    ax = plt.gca()
    ax.add_collection(lc)
    return lc


def make_segments3d(x, y, z):
    """
    Create an array of 3D line segments from x, y, z coordinates.
    Returns an array of shape (num_segments, 2, 3).
    """
    points = np.array([x, y, z]).T  # shape (n, 3)
    segments = np.stack([points[:-1], points[1:]], axis=1)  # shape (n-1, 2, 3)
    return segments


def colorline3d(
    x, y, z, cmap="rainbow", norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0
):
    """
    Plot a colored 3D line with coordinates x, y, z.
    Colors are determined by a normalized array (default linear space from 0 to 1)
    and mapped by cmap.
    Returns the Line3DCollection.
    """
    segments = make_segments3d(x, y, z)
    lc = Line3DCollection(
        segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )
    # Create a normalized array for colors.
    color_array = np.linspace(0.0, 1.0, len(x))
    lc.set_array(color_array)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc


def add_buffer_limit(ax, lower, upper, axis="x"):
    """
    Set the limits for a single axis with a 10% buffer on each side.

    Parameters:
      ax: The matplotlib axis object.
      lower: The lower limit.
      upper: The upper limit.
      axis: Which axis to modify ('x', 'y', or 'z').
    """
    rng = upper - lower
    if rng == 0:
        rng = 1.0
    buf = 0.1 * rng
    if axis == "x":
        ax.set_xlim(lower - buf, upper + buf)
    elif axis == "y":
        ax.set_ylim(lower - buf, upper + buf)
    elif axis == "z":
        ax.set_zlim(lower - buf, upper + buf)


def add_buffer_limits(
    ax, x_min, x_max, y_min, y_max, z_min=None, z_max=None, equal_xy=True
):
    """
    Set axis limits with a 10% buffer on each side.

    If equal_xy is True (default), the limits for x and y are computed using the
    combined range of both axes to enforce an equal aspect ratio (so that a distance
    of 1 in x is equal to a distance of 1 in y). For 3D axes, z_min and z_max are processed
    separately.
    """
    if equal_xy:
        # Compute centers and half-ranges.
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        half_range_x = (x_max - x_min) / 2.0
        half_range_y = (y_max - y_min) / 2.0
        half_range = max(half_range_x, half_range_y)
        if half_range == 0:
            half_range = 0.5
        # Add a 10% buffer to the total range.
        buffer = 0.1 * (2 * half_range)
        new_half_range = half_range + buffer / 2.0
        ax.set_xlim(x_center - new_half_range, x_center + new_half_range)
        ax.set_ylim(y_center - new_half_range, y_center + new_half_range)
    else:
        # x-axis limits.
        x_range = x_max - x_min
        if x_range == 0:
            x_range = 1.0
        x_buffer = 0.1 * x_range
        ax.set_xlim(x_min - x_buffer, x_max + x_buffer)

        # y-axis limits.
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0
        y_buffer = 0.1 * y_range
        ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    # z-axis (if applicable)
    if z_min is not None and z_max is not None:
        z_range = z_max - z_min
        if z_range == 0:
            z_range = 1.0
        z_buffer = 0.1 * z_range
        ax.set_zlim(z_min - z_buffer, z_max + z_buffer)


def set_axes_equal(ax):
    """
    Set the aspect ratio of the 3D plot to be equal.
    This ensures that the sphere appears round.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_middle = (x_limits[0] + x_limits[1]) / 2
    y_middle = (y_limits[0] + y_limits[1]) / 2
    z_middle = (z_limits[0] + z_limits[1]) / 2

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def plot_trajectory_2d(
    position,
    target,
    cmap="rainbow_r",
    figsize=(3, 4.5),
    remove_ticks=True,
    remove_labels=True,
    show_colorbar=False,
    ax_lim: Optional[float] = 1.1,
    position_marker=True,
    target_marker=True,
    base_marker=True,
    linewidth=2,
    on_target_line: Optional[float] = 0.05,
):
    """
    Creates a 2D plot of the trajectory (position) and target trajectory in the top subplot,
    and below it a plot of the Euclidean distance between position and target over time.
    Both lines are drawn as multicolored lines where the color indicates time.
    Axis limits are set to the range of the data plus a 10% buffer.
    If target is stationary, a marker is added.
    Returns the figure and axes.
    """
    # --- Change: Create two subplots instead of one ---
    fig, (ax_traj, ax_dist) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
    )
    T = position.shape[0]
    t = np.linspace(0, 1, T)

    # get the colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # --- Top subplot: Trajectory plot ---
    plt.sca(ax_traj)  # Set current axis to ax_traj.
    lc = colorline2d(
        position[:, 0], position[:, 1], z=t, cmap=cmap, linewidth=linewidth
    )
    colorline2d(target[:, 0], target[:, 1], z=t, cmap=cmap, linewidth=linewidth)

    # Draw original markers with requested colors, symbols, and outlines.
    if base_marker:
        ax_traj.scatter(
            0,
            0,
            color="black",
            s=50,
            marker="P",
            label="Base",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
    if target_marker:
        if on_target_line is not None:
            circle = plt.Circle(
                (target[-1, 0], target[-1, 1]),
                on_target_line,
                facecolor="red",
                fill=True,
                linestyle="-",
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )
            ax_traj.add_artist(circle)
            ax_traj.scatter(
                [],
                [],
                color="red",
                s=50,
                marker="o",
                label="Target",
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )
        else:
            ax_traj.scatter(
                target[-1, 0],
                target[-1, 1],
                color="red",
                s=50,
                marker="o",
                label="Target",
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )
    if position_marker:
        ax_traj.scatter(
            position[0, 0],
            position[0, 1],
            color=cmap(0.0),
            s=100,
            marker="*",
            label="Start Position",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
        ax_traj.scatter(
            position[-1, 0],
            position[-1, 1],
            color=cmap(1.0),
            s=50,
            marker="X",
            label="Final Position",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )

    # Set axis limits for 2D plot.
    if ax_lim is not None:
        ax_traj.set_xlim(-ax_lim, ax_lim)
        ax_traj.set_ylim(-ax_lim, ax_lim)
    else:
        x_min = min(position[:, 0].min(), target[:, 0].min())
        x_max = max(position[:, 0].max(), target[:, 0].max())
        y_min = min(position[:, 1].min(), target[:, 1].min())
        y_max = max(position[:, 1].max(), target[:, 1].max())
        add_buffer_limits(ax_traj, x_min, x_max, y_min, y_max)
    ax_traj.set_aspect("equal")

    if remove_ticks:
        ax_traj.set_xticks([])
        ax_traj.set_yticks([])

    if remove_labels:
        ax_traj.set_xlabel("")
        ax_traj.set_ylabel("")
    else:
        ax_traj.set_xlabel("X")
        ax_traj.set_ylabel("Y")

    if show_colorbar:
        cb = plt.colorbar(lc, ax=ax_traj, ticks=[0, 1], label="Normalized Time")
        cb.ax.set_yticklabels(["Start", "End"])

    ax_traj.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2
    )

    # --- New: Bottom subplot: Distance over time ---
    distance = compute_distance(position, target)
    plt.sca(ax_dist)  # Set current axis to ax_dist.
    if on_target_line is not None:
        # fill the area between 0 and the on_target_line
        ax_dist.fill_between(t, 0, on_target_line, color="red", alpha=0.2)
    colorline2d(t, distance, z=t, cmap=cmap, linewidth=linewidth)

    ax_dist.set_xticks([t[0], t[-1]])
    ax_dist.set_xticklabels(["Start", "End"])
    if remove_ticks:
        ax_dist.set_yticks([])
    ax_dist.set_ylim(0, distance.max() + 0.1)
    ax_dist.set_xlim(0, 1)
    ax_dist.set_xlabel("Time")
    ax_dist.set_ylabel("Distance")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, (ax_traj, ax_dist)


def plot_trajectory_3d(
    position,
    target,
    cmap="rainbow_r",
    figsize=(3, 4.5),
    remove_ticks=True,
    remove_labels=True,
    show_colorbar=False,
    ax_lim: Optional[float] = 0.5,
    show_shadows=True,
    position_marker=True,
    target_marker=True,
    base_marker=True,
    linewidth=2,
    on_target_line: Optional[float] = 0.123,
):
    """
    Creates a 3D plot of the trajectory (position) and target trajectory in the top subplot,
    and below it a 2D plot of the Euclidean distance between position and target over time.
    Both lines are drawn as multicolored lines where the color indicates time.
    Axis limits are set to the range of the data plus a 10% buffer.
    Markers are drawn with the following scheme:
      - Target: red circle ('o')
      - Base: black + marker ('P')
      - Start Position: star ('*')
      - Final Position: thick x ('X')
    All markers have a thin black outline.
    Additionally, each marker is projected onto the three coordinate planes as a shadow,
    drawn in black with alpha=shadow_alpha and no outline.

    Returns the figure and axes (tuple: (ax_traj, ax_dist)).
    """
    T = position.shape[0]
    t = np.linspace(0, 1, T)

    # Create figure with two subplots: top (3D) and bottom (2D).
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax_traj = fig.add_subplot(gs[0], projection="3d")
    ax_dist = fig.add_subplot(gs[1])

    # get the colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # --- Top subplot: 3D Trajectory ---
    plt.sca(ax_traj)
    pos_collection = colorline3d(
        position[:, 0],
        position[:, 1],
        position[:, 2],
        cmap=cmap,
        norm=plt.Normalize(0, 1),
        linewidth=linewidth,
    )
    # plt.sca(ax_traj)
    target_collection = colorline3d(
        target[:, 0],
        target[:, 1],
        target[:, 2],
        cmap=cmap,
        norm=plt.Normalize(0, 1),
        linewidth=linewidth,
    )

    # Draw original markers with requested colors, symbols, and outlines.
    if base_marker:
        ax_traj.scatter(
            0,
            0,
            0,
            color="black",
            s=50,
            marker="P",
            label="Base",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
    if target_marker:
        if on_target_line is not None:
            # Define the sphere parameters
            radius = on_target_line
            center = target[-1, :]

            # Create spherical coordinates
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

            # Plot the surface
            ax_traj.plot_surface(
                x, y, z, color="red", alpha=0.4, linewidth=0, shade=False
            )
            ax_traj.scatter(
                [],
                [],
                [],
                color="red",
                s=50,
                marker="o",
                label="Target",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
                zorder=2,
            )
        else:
            ax_traj.scatter(
                target[-1, 0],
                target[-1, 1],
                target[-1, 2],
                color="red",
                s=50,
                marker="o",
                label="Target",
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )
    if position_marker:
        ax_traj.scatter(
            position[0, 0],
            position[0, 1],
            position[0, 2],
            color=cmap(0.0),
            s=100,
            marker="*",
            label="Start Position",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
        ax_traj.scatter(
            position[-1, 0],
            position[-1, 1],
            position[-1, 2],
            color=cmap(1.0),
            s=50,
            marker="X",
            label="Final Position",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )

    # Set axis limits.
    z_min = min(position[:, 2].min(), target[:, 2].min(), 0)
    z_max = max(position[:, 2].max(), target[:, 2].max())
    if ax_lim is not None:
        ax_traj.set_xlim(-ax_lim, ax_lim)
        ax_traj.set_ylim(-ax_lim, ax_lim)
        add_buffer_limit(ax_traj, z_min, z_max, axis="z")
    else:
        x_min = min(position[:, 0].min(), target[:, 0].min())
        x_max = max(position[:, 0].max(), target[:, 0].max())
        y_min = min(position[:, 1].min(), target[:, 1].min())
        y_max = max(position[:, 1].max(), target[:, 1].max())
        # make sure the x and y axis are centered at 0
        x_min = min(x_min, -x_max)
        x_max = max(x_max, -x_min)
        y_min = min(y_min, -y_max)
        y_max = max(y_max, -y_min)
        # add a buffer to the axis limits
        add_buffer_limits(ax_traj, x_min, x_max, y_min, y_max, z_min, z_max)

    labelpad = 0
    if remove_ticks:
        ax_traj.set_xticklabels([])
        ax_traj.set_yticklabels([])
        ax_traj.set_zticklabels([])
        labelpad = -10
    if remove_labels:
        ax_traj.set_xlabel("")
        ax_traj.set_ylabel("")
        ax_traj.set_zlabel("")
    else:
        ax_traj.set_xlabel("X", labelpad=labelpad)
        ax_traj.set_ylabel("Y", labelpad=labelpad)
        ax_traj.set_zlabel("Z", labelpad=labelpad)

    # Set equal aspect ratio for all axes
    set_axes_equal(ax_traj)
    ax_traj.set_box_aspect([1, 1, 1])

    # Set the color of the axes panes to white.
    ax_traj.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # RGBA for white
    ax_traj.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_traj.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    if show_colorbar:
        cb = plt.colorbar(
            pos_collection, ax=ax_traj, ticks=[0, 1], pad=0.1, label="Normalized Time"
        )
        cb.ax.set_yticklabels(["Start", "End"])

    ax_traj.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)

    # --- New: Add shadow projections ("shadows") for each marker ---
    if show_shadows:
        # Obtain projection coordinates.
        x_low = ax_traj.get_xlim()[0]
        y_high = ax_traj.get_ylim()[1]
        z_low = ax_traj.get_zlim()[0]
        shadow_edge = "#A9A9A9"  # Slightly lighter shadow color for better visibility
        # Static shadows for base and start markers (outline only).
        if base_marker:
            ax_traj.scatter(
                0,
                0,
                z_low,
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="P",
                depthshade=False,
            )
            ax_traj.scatter(
                0,
                y_high,
                0,
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="P",
                depthshade=False,
            )
            ax_traj.scatter(
                x_low,
                0,
                0,
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="P",
                depthshade=False,
            )
        if position_marker:
            ax_traj.scatter(
                position[0, 0],
                position[0, 1],
                z_low,
                facecolors="none",
                edgecolors=shadow_edge,
                s=100,
                marker="*",
                depthshade=False,
            )
            ax_traj.scatter(
                position[0, 0],
                y_high,
                position[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=100,
                marker="*",
                depthshade=False,
            )
            ax_traj.scatter(
                x_low,
                position[0, 1],
                position[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=100,
                marker="*",
                depthshade=False,
            )
        # Dynamic shadows for target and final markers (outline only).
        if target_marker:
            shadow_xy_target = ax_traj.scatter(
                target[0, 0],
                target[0, 1],
                z_low,
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="o",
                depthshade=False,
            )
            shadow_xz_target = ax_traj.scatter(
                target[0, 0],
                y_high,
                target[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="o",
                depthshade=False,
            )
            shadow_yz_target = ax_traj.scatter(
                x_low,
                target[0, 1],
                target[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="o",
                depthshade=False,
            )
        else:
            shadow_xy_target = shadow_xz_target = shadow_yz_target = None
        if position_marker:
            shadow_xy_final = ax_traj.scatter(
                position[0, 0],
                position[0, 1],
                z_low,
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="X",
                depthshade=False,
            )
            shadow_xz_final = ax_traj.scatter(
                position[0, 0],
                y_high,
                position[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="X",
                depthshade=False,
            )
            shadow_yz_final = ax_traj.scatter(
                x_low,
                position[0, 1],
                position[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="X",
                depthshade=False,
            )
        else:
            shadow_xy_final = shadow_xz_final = shadow_yz_final = None
    else:
        shadow_xy_target = shadow_xz_target = shadow_yz_target = None
        shadow_xy_final = shadow_xz_final = shadow_yz_final = None

    # --- Bottom subplot: Distance over time ---
    distance = compute_distance(position, target)
    plt.sca(ax_dist)
    # fill between 0 and on_target_line
    if on_target_line is not None:
        ax_dist.fill_between(t, 0, on_target_line, color="red", alpha=0.2)
    colorline2d(t, distance, z=t, cmap=cmap, linewidth=linewidth)
    ax_dist.set_xticks([t[0], t[-1]])
    ax_dist.set_xticklabels(["Start", "End"])
    if remove_ticks:
        ax_dist.set_yticks([])
    ax_dist.set_ylim(0, distance.max() + 0.1)
    ax_dist.set_xlim(0, 1)
    ax_dist.set_xlabel("Time")
    ax_dist.set_ylabel("Distance")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    # Adjust subplots manually.
    fig.subplots_adjust(
        left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.1, wspace=0.1
    )

    return fig, (ax_traj, ax_dist)


def plot_trajectory(position, target, cmap="rainbow_r", figsize=(7, 6), **kwargs):
    """
    Plots a trajectory (position) and target trajectory.
    If the input arrays have 2 columns, a 2D plot is created;
    if they have 3 columns, a 3D plot is created.

    Parameters:
      - position: Array of shape (T, d) with d = 2 or 3.
      - target: Array of shape (T, d) with the same d as position.
      - cmap: Colormap to use (default 'rainbow').

    Returns:
      The matplotlib figure and axis objects.
    """
    position = np.asarray(position)
    target = np.asarray(target)
    if position.shape[1] != target.shape[1]:
        raise ValueError("Position and target must have the same spatial dimension.")
    if position.shape[1] == 2:
        return plot_trajectory_2d(
            position, target, cmap=cmap, figsize=figsize, **kwargs
        )
    elif position.shape[1] == 3:
        return plot_trajectory_3d(
            position, target, cmap=cmap, figsize=figsize, **kwargs
        )
    else:
        raise ValueError("Only 2D or 3D trajectories are supported.")


# -------------------------------
# Animation Functions (Stable Colors)
# -------------------------------


def animate_trajectory_2d(
    position,
    target,
    cmap="rainbow_r",
    fps: int = 10,
    skip: int = 1,
    figsize=(3, 4.5),
    remove_ticks: bool = True,
    show_colorbar: bool = False,
    remove_labels: bool = True,
    ax_lim: Optional[float] = 1.1,
    position_marker: bool = True,
    target_marker: bool = True,
    base_marker: bool = True,
    linewidth: int = 2,
    on_target_line: Optional[float] = 0.05,
) -> FuncAnimation:
    """
    Animates a 2D trajectory and target along with a lower subplot showing the Euclidean
    distance between them over time. At each frame the markers (target and final position)
    update to reflect the current time step.

    Returns the animation object.
    """
    # Create two subplots: top for the trajectory, bottom for the distance plot.
    fig, (ax_traj, ax_dist) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
    )
    T = position.shape[0]
    t = np.linspace(0, 1, T)

    # Get the colormap.
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Precompute full segments and fixed color arrays for the trajectory.
    full_pos_segments = make_segments(position[:, 0], position[:, 1])
    full_tar_segments = make_segments(target[:, 0], target[:, 1])
    full_pos_colors = np.linspace(0, 1, len(full_pos_segments))
    full_tar_colors = np.linspace(0, 1, len(full_tar_segments))

    # Precompute the distance and its segments.
    distance = compute_distance(position, target)
    full_dist_segments = make_segments(t, distance)
    full_dist_colors = np.linspace(0, 1, len(full_dist_segments))

    # Create empty LineCollections for the trajectory (top subplot)...
    pos_collection = mcoll.LineCollection(
        [], cmap=cmap, norm=plt.Normalize(0, 1), linewidth=linewidth, alpha=1.0
    )
    tar_collection = mcoll.LineCollection(
        [], cmap=cmap, norm=plt.Normalize(0, 1), linewidth=linewidth, alpha=1.0
    )
    ax_traj.add_collection(pos_collection)
    ax_traj.add_collection(tar_collection)

    # ...and for the distance plot (bottom subplot).
    dist_collection = mcoll.LineCollection(
        [], cmap=cmap, norm=plt.Normalize(0, 1), linewidth=linewidth, alpha=1.0
    )
    ax_dist.add_collection(dist_collection)

    # Draw original markers with requested colors, symbols, and outlines.
    scatter_final, scatter_target = None, None
    if base_marker:
        ax_traj.scatter(
            0,
            0,
            color="black",
            s=50,
            marker="P",
            label="Base",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
    if target_marker:
        scatter_target = ax_traj.scatter(
            target[0, 0],
            target[0, 1],
            color="red",
            s=50,
            marker="o",
            label="Target",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
    if position_marker:
        ax_traj.scatter(
            position[0, 0],
            position[0, 1],
            color=cmap(0.0),
            s=100,
            marker="*",
            label="Start Position",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
        scatter_final = ax_traj.scatter(
            position[0, 0],
            position[0, 1],
            color=cmap(1.0),
            s=50,
            marker="X",
            label="Position",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )

    # Set axis limits for the trajectory subplot.
    if ax_lim is not None:
        ax_traj.set_xlim(-ax_lim, ax_lim)
        ax_traj.set_ylim(-ax_lim, ax_lim)
    else:
        x_min = min(position[:, 0].min(), target[:, 0].min())
        x_max = max(position[:, 0].max(), target[:, 0].max())
        y_min = min(position[:, 1].min(), target[:, 1].min())
        y_max = max(position[:, 1].max(), target[:, 1].max())
        add_buffer_limits(ax_traj, x_min, x_max, y_min, y_max)
    ax_traj.set_aspect("equal")

    if remove_ticks:
        ax_traj.set_xticks([])
        ax_traj.set_yticks([])

    if remove_labels:
        ax_traj.set_xlabel("")
        ax_traj.set_ylabel("")
    else:
        ax_traj.set_xlabel("X")
        ax_traj.set_ylabel("Y")

    if show_colorbar:
        cb = plt.colorbar(
            pos_collection, ax=ax_traj, ticks=[0, 1], label="Normalized Time"
        )
        cb.ax.set_yticklabels(["Start", "End"])

    # Update legend: no frame, centered above, 2 columns.
    ax_traj.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2
    )

    # Set axis limits and labels for the distance subplot.
    if on_target_line is not None:
        ax_dist.fill_between(t, 0, on_target_line, color="red", alpha=0.2)
    ax_dist.set_xticks([t[0], t[-1]])
    ax_dist.set_xticklabels(["Start", "End"])
    if remove_ticks:
        ax_dist.set_yticks([])
    ax_dist.set_ylim(0, distance.max() + 0.1)
    ax_dist.set_xlim(0, 1)
    ax_dist.set_xlabel("Time")
    ax_dist.set_ylabel("Distance")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    plt.tight_layout()

    # Build frame indices that skip frames but include the last frame.
    frames = np.arange(0, T, skip)
    if frames[-1] != T - 1:
        frames = np.append(frames, T - 1)

    def update(frame):
        if frame < 1:
            pos_collection.set_segments([])
            tar_collection.set_segments([])
            dist_collection.set_segments([])
        else:
            # Update line segments.
            pos_collection.set_segments(full_pos_segments[:frame])
            pos_collection.set_array(full_pos_colors[:frame])
            tar_collection.set_segments(full_tar_segments[:frame])
            tar_collection.set_array(full_tar_colors[:frame])
            dist_collection.set_segments(full_dist_segments[:frame])
            dist_collection.set_array(full_dist_colors[:frame])
            # Update markers to reflect current time step.
            if target_marker and scatter_target is not None:
                scatter_target.set_offsets(
                    np.array([[target[frame, 0], target[frame, 1]]])
                )
            if position_marker and scatter_final is not None:
                scatter_final.set_offsets(
                    np.array([[position[frame, 0], position[frame, 1]]])
                )
        return (
            [
                pos_collection,
                tar_collection,
                dist_collection,
                scatter_target,
                scatter_final,
            ]
            if (target_marker or position_marker)
            else [pos_collection, tar_collection, dist_collection]
        )

    anim = FuncAnimation(
        fig, update, frames=frames, interval=1000 / fps, blit=True, repeat=False
    )
    plt.close()
    return anim


def animate_trajectory_3d(
    position,
    target,
    cmap="rainbow_r",
    fps: int = 10,
    skip: int = 1,
    figsize=(3, 4.5),
    remove_ticks=True,
    remove_labels=True,
    show_colorbar=False,
    show_shadows=True,
    ax_lim: Optional[float] = 0.5,
    base_marker=True,
    target_marker=True,
    position_marker=True,
    linewidth=2,
    on_target_line: Optional[float] = 0.123,
) -> FuncAnimation:
    """
    Animates a 3D trajectory and target along with a lower subplot showing the Euclidean
    distance between them over time. Supports the same marker scheme as the static plot:
        - Target: red circle ('o')
        - Base: black + marker ('P')
        - Start Position: star ('*')
        - Final Position: thick x ('X')
    All markers have a thin black outline (the final marker has a thicker outline).
    If show_shadows is True, each marker projects its shadow (in black, alpha=shadow_alpha, no outline)
    onto the XY-, XZ-, and YZ-planes.

    Returns the animation object.
    """

    T = position.shape[0]
    t = np.linspace(0, 1, T)

    # Create figure with two subplots: top (3D) and bottom (2D).
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax_traj = fig.add_subplot(gs[0], projection="3d")
    ax_dist = fig.add_subplot(gs[1])

    # Get the colormap.
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Precompute full segments and fixed color arrays for the 3D trajectory.
    full_pos_segments = make_segments3d(position[:, 0], position[:, 1], position[:, 2])
    full_tar_segments = make_segments3d(target[:, 0], target[:, 1], target[:, 2])
    full_pos_colors = np.linspace(0, 1, len(full_pos_segments))
    full_tar_colors = np.linspace(0, 1, len(full_tar_segments))

    # Precompute the distance and its 2D segments.
    distance = compute_distance(position, target)
    full_dist_segments = make_segments(t, distance)
    full_dist_colors = np.linspace(0, 1, len(full_dist_segments))

    # Create empty LineCollections for the 3D trajectory.
    pos_collection = Line3DCollection(
        [], cmap=cmap, norm=plt.Normalize(0, 1), linewidth=linewidth, alpha=1.0
    )
    tar_collection = Line3DCollection(
        [], cmap=cmap, norm=plt.Normalize(0, 1), linewidth=linewidth, alpha=1.0
    )
    ax_traj.add_collection(pos_collection)
    ax_traj.add_collection(tar_collection)

    # Create empty LineCollection for the distance plot.
    dist_collection = LineCollection(
        [], cmap=cmap, norm=plt.Normalize(0, 1), linewidth=linewidth, alpha=1.0
    )
    ax_dist.add_collection(dist_collection)

    # --- Markers ---
    # Static markers (base and start) remain fixed.
    if base_marker:
        base_sc = ax_traj.scatter(
            0,
            0,
            0,
            color="black",
            s=50,
            marker="P",
            label="Base",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
    if target_marker:
        if on_target_line is not None:
            # Define the sphere parameters
            radius = on_target_line
            center = target[-1, :]

            # Create spherical coordinates
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

            # Plot the surface
            target_sc = ax_traj.plot_surface(
                x, y, z, color="red", alpha=0.4, linewidth=0, shade=False
            )
            ax_traj.scatter(
                [],
                [],
                [],
                color="red",
                s=50,
                marker="o",
                label="Target",
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )
        else:
            target_sc = ax_traj.scatter(
                target[0, 0],
                target[0, 1],
                target[0, 2],
                color="red",
                s=50,
                marker="o",
                label="Target",
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )
    else:
        target_sc = None
    if position_marker:
        start_sc = ax_traj.scatter(
            position[0, 0],
            position[0, 1],
            position[0, 2],
            color=cmap(0.0),
            s=100,
            marker="*",
            label="Start Position",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
        final_sc = ax_traj.scatter(
            position[0, 0],
            position[0, 1],
            position[0, 2],
            color=cmap(1.0),
            s=50,
            marker="X",
            label="Position",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
    else:
        final_sc = None

    # --- Axis Limits ---
    z_min = min(position[:, 2].min(), target[:, 2].min(), 0)
    z_max = max(position[:, 2].max(), target[:, 2].max())
    if ax_lim is not None:
        ax_traj.set_xlim(-ax_lim, ax_lim)
        ax_traj.set_ylim(-ax_lim, ax_lim)
        add_buffer_limit(ax_traj, z_min, z_max, axis="z")
    else:
        x_min = min(position[:, 0].min(), target[:, 0].min())
        x_max = max(position[:, 0].max(), target[:, 0].max())
        y_min = min(position[:, 1].min(), target[:, 1].min())
        y_max = max(position[:, 1].max(), target[:, 1].max())
        # make sure the x and y axis are centered at 0
        x_min = min(x_min, -x_max)
        x_max = max(x_max, -x_min)
        y_min = min(y_min, -y_max)
        y_max = max(y_max, -y_min)
        # add a buffer to the axis limits
        add_buffer_limits(ax_traj, x_min, x_max, y_min, y_max, z_min, z_max)

    labelpad = 0
    if remove_ticks:
        ax_traj.set_xticklabels([])
        ax_traj.set_yticklabels([])
        ax_traj.set_zticklabels([])
        labelpad = -10

    if remove_labels:
        ax_traj.set_xlabel("")
        ax_traj.set_ylabel("")
        ax_traj.set_zlabel("")
    else:
        ax_traj.set_xlabel("X", labelpad=labelpad)
        ax_traj.set_ylabel("Y", labelpad=labelpad)
        ax_traj.set_zlabel("Z", labelpad=labelpad)

    # Set equal aspect ratio for all axes
    set_axes_equal(ax_traj)
    ax_traj.set_box_aspect([1, 1, 1])

    # Set the color of the axes panes to white.
    ax_traj.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # RGBA for white
    ax_traj.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_traj.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    if show_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = plt.colorbar(
            sm, ax=ax_traj, ticks=[0, 1], pad=0.1, label="Normalized Time"
        )
        cb.ax.set_yticklabels(["Start", "End"])

    ax_traj.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)

    # --- Shadows ---
    if show_shadows:
        # Obtain projection coordinates.
        x_low = ax_traj.get_xlim()[0]
        y_high = ax_traj.get_ylim()[1]
        z_low = ax_traj.get_zlim()[0]
        shadow_edge = "#A9A9A9"  # Slightly lighter shadow color for better visibility
        # Static shadows for base and start markers (outline only).
        if base_marker:
            ax_traj.scatter(
                0,
                0,
                z_low,
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="P",
                depthshade=False,
            )
            ax_traj.scatter(
                0,
                y_high,
                0,
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="P",
                depthshade=False,
            )
            ax_traj.scatter(
                x_low,
                0,
                0,
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="P",
                depthshade=False,
            )
        if position_marker:
            ax_traj.scatter(
                position[0, 0],
                position[0, 1],
                z_low,
                facecolors="none",
                edgecolors=shadow_edge,
                s=100,
                marker="*",
                depthshade=False,
            )
            ax_traj.scatter(
                position[0, 0],
                y_high,
                position[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=100,
                marker="*",
                depthshade=False,
            )
            ax_traj.scatter(
                x_low,
                position[0, 1],
                position[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=100,
                marker="*",
                depthshade=False,
            )
        # Dynamic shadows for target and final markers (outline only).
        if target_marker:
            shadow_xy_target = ax_traj.scatter(
                target[0, 0],
                target[0, 1],
                z_low,
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="o",
                depthshade=False,
            )
            shadow_xz_target = ax_traj.scatter(
                target[0, 0],
                y_high,
                target[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="o",
                depthshade=False,
            )
            shadow_yz_target = ax_traj.scatter(
                x_low,
                target[0, 1],
                target[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="o",
                depthshade=False,
            )
        else:
            shadow_xy_target = shadow_xz_target = shadow_yz_target = None
        if position_marker:
            shadow_xy_final = ax_traj.scatter(
                position[0, 0],
                position[0, 1],
                z_low,
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="X",
                depthshade=False,
            )
            shadow_xz_final = ax_traj.scatter(
                position[0, 0],
                y_high,
                position[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="X",
                depthshade=False,
            )
            shadow_yz_final = ax_traj.scatter(
                x_low,
                position[0, 1],
                position[0, 2],
                facecolors="none",
                edgecolors=shadow_edge,
                s=50,
                marker="X",
                depthshade=False,
            )
        else:
            shadow_xy_final = shadow_xz_final = shadow_yz_final = None
    else:
        shadow_xy_target = shadow_xz_target = shadow_yz_target = None
        shadow_xy_final = shadow_xz_final = shadow_yz_final = None

    # --- Bottom subplot: Distance over time ---
    # fill between 0 and on_target_line
    if on_target_line is not None:
        ax_dist.fill_between(t, 0, on_target_line, color="red", alpha=0.2)
    ax_dist.set_xticks([t[0], t[-1]])
    ax_dist.set_xticklabels(["Start", "End"])
    if remove_ticks:
        ax_dist.set_yticks([])
    ax_dist.set_ylim(0, distance.max() + 0.1)
    ax_dist.set_xlim(0, 1)
    ax_dist.set_xlabel("Time")
    ax_dist.set_ylabel("Distance")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.1, wspace=0.1
    )

    # --- Build frame indices ---
    frames_arr = np.arange(0, T, skip)
    if frames_arr[-1] != T - 1:
        frames_arr = np.append(frames_arr, T - 1)

    def update(frame):
        if frame < 1:
            pos_collection.set_segments([])
            tar_collection.set_segments([])
            dist_collection.set_segments([])
        else:
            pos_collection.set_segments(full_pos_segments[:frame])
            pos_collection.set_array(full_pos_colors[:frame])
            tar_collection.set_segments(full_tar_segments[:frame])
            tar_collection.set_array(full_tar_colors[:frame])
            dist_collection.set_segments(full_dist_segments[:frame])
            dist_collection.set_array(full_dist_colors[:frame])

            # Update dynamic markers.
            if target_marker and target_sc is not None:
                target_sc._offsets3d = (
                    [target[frame, 0]],
                    [target[frame, 1]],
                    [target[frame, 2]],
                )
            if position_marker and final_sc is not None:
                final_sc._offsets3d = (
                    [position[frame, 0]],
                    [position[frame, 1]],
                    [position[frame, 2]],
                )

            # Update dynamic shadows.
            if show_shadows and target_marker and shadow_xy_target is not None:
                shadow_xy_target._offsets3d = (
                    [target[frame, 0]],
                    [target[frame, 1]],
                    [z_low],
                )
                shadow_xz_target._offsets3d = (
                    [target[frame, 0]],
                    [y_high],
                    [target[frame, 2]],
                )
                shadow_yz_target._offsets3d = (
                    [x_low],
                    [target[frame, 1]],
                    [target[frame, 2]],
                )
            if show_shadows and position_marker and shadow_xy_final is not None:
                shadow_xy_final._offsets3d = (
                    [position[frame, 0]],
                    [position[frame, 1]],
                    [z_low],
                )
                shadow_xz_final._offsets3d = (
                    [position[frame, 0]],
                    [y_high],
                    [position[frame, 2]],
                )
                shadow_yz_final._offsets3d = (
                    [x_low],
                    [position[frame, 1]],
                    [position[frame, 2]],
                )

        artists = [pos_collection, tar_collection, dist_collection]
        if target_marker:
            artists.append(target_sc)
        if position_marker:
            artists.append(final_sc)
        if show_shadows and target_marker:
            artists.extend([shadow_xy_target, shadow_xz_target, shadow_yz_target])
        if show_shadows and position_marker:
            artists.extend([shadow_xy_final, shadow_xz_final, shadow_yz_final])
        return artists

    anim = FuncAnimation(
        fig, update, frames=frames_arr, interval=1000 / fps, blit=False, repeat=False
    )
    plt.close()
    return anim


def animate_trajectory(
    position, target, cmap="rainbow_r", fps=10, skip=1, figsize=(4, 5), **kwargs
):
    """
    Wrapper that selects the 2D or 3D animation based on the input array dimensions.
    """
    position = np.asarray(position)
    target = np.asarray(target)
    if position.shape[1] != target.shape[1]:
        raise ValueError("Position and target must have the same spatial dimension.")
    if position.shape[1] == 2:
        return animate_trajectory_2d(
            position, target, cmap=cmap, fps=fps, skip=skip, figsize=figsize, **kwargs
        )
    elif position.shape[1] == 3:
        return animate_trajectory_3d(
            position, target, cmap=cmap, fps=fps, skip=skip, figsize=figsize, **kwargs
        )
    else:
        raise ValueError("Only 2D or 3D trajectories are supported.")


class TrajectoryPlotter:
    """
    A trajectory plotter class that extracts and plots the positions of the end effector of a robot arm as well as the target positions.

    """

    def __init__(
        self,
        env,
        plot: Union[bool, int, List[int]] = True,
        animate: Union[bool, int, List[int]] = True,
        cmap: Optional[str] = None,
        figsize: tuple = (3, 4.5),
        skip: int = 1,
        max_trajectories: int = 4,
    ):
        self.trajectories = []
        self.max_trajectories = max_trajectories

        if not isinstance(plot, (bool, list)):
            assert isinstance(
                plot, (int, np.integer)
            ), "The plot attribute must be a boolean, an integer, or a list of integers."
            plot = [plot]
        if not isinstance(animate, (bool, list)):
            assert isinstance(
                animate, (int, np.integer)
            ), "The animate attribute must be a boolean, an integer, or a list of integers."
            animate = [animate]
        assert isinstance(
            plot, (bool, list)
        ), "The plot attribute must be a boolean or a list of integers."
        assert isinstance(
            animate, (bool, list)
        ), "The animate attribute must be a boolean or a list of integers."
        self.plot = plot
        self.animate = animate
        self.figsize = figsize
        self.skip = skip

        if cmap is None:
            # Define the two colors (as hex or RGB)
            colors = ["#87cc6e", "#25499c"]
            # Create a custom linear colormap
            cmap = LinearSegmentedColormap.from_list("custom_blend", colors)
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        self.cmap = cmap

        # get some info from the environment where to find the hand position in the state observations
        loss_gain = env.unwrapped.call("get_loss_gain")[0]

        if not isinstance(loss_gain, dict):
            raise ValueError("The loss_gain attribute is not a dictionary.")
        # then we need to check that the loss gain dictionary has a use key
        if "use" not in loss_gain:
            raise ValueError("The loss_gain dictionary does not have a 'use' key.")
        # finally we need to check that the use key is an iterable of integers
        if not isinstance(loss_gain["use"], Iterable):
            raise ValueError("The loss_gain['use'] key is not an iterable object.")
        if not all(isinstance(i, (int, np.integer)) for i in loss_gain["use"]):
            raise ValueError("The loss_gain['use'] key is not a list of integers.")
        # finally need to check if the list has 2 or 3 elements (2 for 2D and 3 for 3D)
        if len(loss_gain["use"]) not in [2, 3]:
            raise ValueError("The loss_gain['use'] key is not a list of length 2 or 3.")
        self.hand_idx = loss_gain["use"]

        # also calculate the fps of the environment
        self.fps = env.unwrapped.metadata["render_fps"]

    def __call__(self, episodes: Union[Episode, List[Episode]]):
        if isinstance(episodes, Episode):
            episodes = [episodes]
        for episode in episodes:
            self.extract(episode)
            if len(self.trajectories) >= self.max_trajectories:
                break
        print(f"Extracted total number of trajectories: {len(self.trajectories)}")
        if self.plot:
            figures = self.plot_trajectories()
        if self.animate:
            animations = self.animate_trajectories()
        return figures, animations

    def extract(self, episode):
        hand_positions = []
        target_positions = []
        for prediction in episode:
            hand_positions.append(
                prediction.state[self.hand_idx].detach().cpu().numpy()
            )
            target_positions.append(prediction.target.detach().cpu().numpy())
        hand_positions = np.stack(hand_positions, axis=0)
        target_positions = np.stack(target_positions, axis=0)
        trajectory = {"position": hand_positions, "target": target_positions}
        self.trajectories.append(trajectory)

    def plot_trajectory(self, idx: int):

        trajectory = self.trajectories[idx]
        fig, _ = plot_trajectory(
            position=trajectory["position"],
            target=trajectory["target"],
            cmap=self.cmap,
            figsize=self.figsize,
        )
        return fig

    def plot_trajectories(self):
        figures = []
        if isinstance(self.plot, bool) and self.plot:
            # plot all trajectories
            for idx in range(len(self.trajectories)):
                fig = self.plot_trajectory(idx)
                figures.append(fig)
        elif isinstance(self.plot, list):
            for idx in self.plot:
                fig = self.plot_trajectory(idx)
                figures.append(fig)
        return figures

    def animate_trajectory(self, idx: int):
        trajectory = self.trajectories[idx]
        anim = animate_trajectory(
            position=trajectory["position"],
            target=trajectory["target"],
            cmap=self.cmap,
            fps=self.fps,
            skip=self.skip,
            figsize=self.figsize,
        )
        return anim

    def animate_trajectories(self):
        animations = []
        if isinstance(self.animate, bool) and self.animate:
            # animate all trajectories
            for idx in range(len(self.trajectories)):
                anim = self.animate_trajectory(idx)
                animations.append(anim)
        elif isinstance(self.animate, list):
            for idx in self.animate:
                anim = self.animate_trajectory(idx)
                animations.append(anim)
        return animations
