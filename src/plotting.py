import numpy as np
import matplotlib as mlp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from .extratypes import *
from .utils import FrameStack


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