import torch
import numpy as np
import matplotlib as mlp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from .extratypes import *
from .utils import FrameStack
from .eval_helpers import make_predictions
from .memory import Episode
from celluloid import Camera


def render_video(
        framestacks: list[FrameStack], 
        framerate: int = 30, 
        dpi: int = 70,
        save: Optional[Union[Path, str]] = '',
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

    orig_backend = mlp.get_backend()
    mlp.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    mlp.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(
        fig=fig, 
        func=update, 
        frames=frames,
        interval=interval, 
        blit=True, 
        repeat=False
    )
    if save:
        print(f"Saving animation to {save}")
        anim.save(save)

    return anim


def animate_predictions(
    episodes: list[Episode],
    transition_model: torch.nn.Module,
    labels: list[str],
    unroll: int = 1,
    warmup: int = 0,
    step: int = 10,
    deterministic: bool = True,
    fps: int = 30,
    dpi: int = 50,
    font_size: int = 12,
    save: Optional[Union[Path, str]] = '',
    max_animations: int = 1,
) -> object:
    
    predictions = make_predictions(
        transition_model=transition_model,
        episodes=episodes,
        warmup=warmup,
        unroll=unroll,
        step=step,
        deterministic=deterministic,
    )
    predictions = predictions.detach().cpu().numpy()
    next_states = torch.stack([torch.stack([s.next_state for s in episode]) for episode in episodes], dim=1).cpu().numpy()

    N = predictions.shape[2]

    animations = []

    for i in range(min(N, max_animations)):
        animation = animate_prediction(predictions[:, :, i], next_states[:, :, i], labels, warmup, step, fps, dpi, font_size, save)
        animations.append(animation)

    return animations


def animate_prediction(
    predictions: np.ndarray,
    next_states: np.ndarray,
    labels: list[str],
    warmup: int = 0,
    step: int = 10,
    fps: int = 30,
    dpi: int = 50,
    font_size: int = 12,
    save: Optional[Union[Path, str]] = '',
) -> object:
    """animate predictions and save to disk
    Args:
        predictions: array of shape [T, unroll, dim]
        next_states: array of shape [T, unroll, dim]
        labels: list of labels for each dimension
        warmup: number of steps to warmup the model
        fps: frames per second
        dpi: dots per inch
        font_size: font size for the legend
        save: path to save the video to

    Returns:
        animation object    
    """

    plt.rcParams['font.size'] = f'{font_size}'

    T, h, D = predictions.shape
    T = T * step

    fig, ax = plt.subplots(D, figsize=(5, D), sharex=True, sharey=True, dpi=dpi)
    plt.ylim(-1.1, 1.1)

    cmap = plt.get_cmap('plasma')

    camera = Camera(fig)

    # make an initial snapshot without prediction
    for d in range(D):
        ax[d].plot([o[d] for o in next_states], c='g', alpha=0.5)
        ax[d].set_ylabel(labels[d])
    ax[-1].set_xlabel('step')
    plt.tight_layout()

    camera.snap()
    idx = np.arange(0., 1., 1. / (h))

    # animate the prediction
    for i, t in enumerate(np.arange(0, T, step)):
        for d in range(D):
            max_ = np.min([h, T - t])
            ax[d].scatter(np.arange(t, np.min([t + warmup, T])), predictions[i, :np.min([warmup, T - t]), d], c='k', s=4)
            if T - t > warmup:
                ax[d].scatter(np.arange(t + warmup, np.min([t + h, T])), predictions[i, warmup:max_, d], c=cmap(idx[:max_-warmup]), s=4)
            ax[d].plot([o[d] for o in next_states], c='g', alpha=0.5)

        plt.tight_layout()
        camera.snap()

    animation = camera.animate(interval=step * 1000. / fps, blit=True)
    plt.close()

    if save:
        print(f'Saving animation to {save}')
        animation.save(save) #, bitrate=-1)

    return animation