import wandb
from .extratypes import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from rich.pretty import pprint
from rich.console import Console
import pandas as pd
from pathlib import Path
import time
from src.utils import conf_to_dict
import logging

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class BaseLogger:
    """base logger class"""

    def __init__(
        self,
    ) -> None:
        self.step_ = 0

    def step(self) -> None:
        """increment the step counter"""
        self.step_ += 1

    def set_step(self, step: int) -> None:
        """set the step counter
        Args:
            step (int): step

        Raises:
            AssertionError: if step is not an int
            AssertionError: if step is less than the current step

        Returns:
            None
        """
        assert isinstance(step, int), "step must be an int"
        assert (
            step >= self.step_
        ), "step must be greater than or equal to the current step"

        self.step_ = step

    def get_step(self) -> int:
        """get the step counter"""
        return self.step_

    def log(
        self, data: dict, prefix: Optional[str] = None, step: Optional[int] = None
    ) -> None:
        """log a dictionary
        Args:
            data (dict): data
            prefix (Optional[str], optional): prefix. Defaults to None.
            step (Optional[int], optional): step. Defaults to None.

        Raises:
            NotImplementedError: if not implemented in subclass

        Returns:
            None
        """
        raise NotImplementedError

    def is_value(self, val) -> bool:
        """detect if the value is int, float, str, or bool
        Args:
            val (any): value

        Returns:
            bool: True if the value is int, float, str, or bool, else False
        """

        return isinstance(val, (int, float, str, bool, np.number))

    def is_figure(self, val) -> bool:
        """detect if the value is a matplotlib figure or an image
        Args:
            val (any): value

        Returns:
            bool: True if the value is a matplotlib figure or an numpy array in image shape, else False
        """

        return (
            isinstance(val, plt.Figure)
            or isinstance(val, np.ndarray)
            and len(val.shape) == 3
        )

    def is_video(self, val) -> bool:
        """detect if the value is a matplotlib animation or a video
        Args:
            val (any): value

        Returns:
            bool: True if the value is a matplotlib animation or a video, else False
        """

        return (
            isinstance(val, animation.FuncAnimation)
            or isinstance(val, animation.ArtistAnimation)
            or isinstance(val, np.ndarray)
            and len(val.shape) == 4
        )

    def is_media(self, val) -> bool:
        """detect if the value is a matplotlib figure, animation, or an image
        Args:
            val (any): value

        Returns:
            bool: True if the value is a matplotlib figure, animation, or an image, else False
        """

        return self.is_figure(val) or self.is_video(val)

    def is_array(self, val) -> bool:
        """detect if the value is a numpy array
        Args:
            val (any): value

        Returns:
            bool: True if the value is a numpy array, else False
        """

        return isinstance(val, np.ndarray)

    def is_tensor(self, val) -> bool:
        """detect if the value is a torch tensor
        Args:
            val (any): value

        Returns:
            bool: True if the value is a torch tensor, else False
        """

        return isinstance(val, torch.Tensor)


class WandBLogger(BaseLogger):
    """wandb logger class"""

    def __init__(
        self,
        run: wandb.run,
    ) -> None:
        super().__init__()
        self.run = run

    def log(
        self, data: dict, prefix: Optional[str] = None, step: Optional[int] = None
    ) -> None:
        """log a dictionary to wandb
        Args:
            data (dict): data
            prefix (Optional[str], optional): prefix. Defaults to None.
            step (Optional[int], optional): step. Defaults to None.

        Raises:
            ValueError: if the value is not of type dict

        Returns:
            None
        """

        assert isinstance(data, dict), "data must be a dict"

        if step is not None:
            self.set_step(step)

        for key, val in data.items():
            if prefix is not None:
                key = f"{prefix}/{key}"
            # if the value is a dict, log it recursively
            if isinstance(val, dict):
                self.log(val, prefix=key)
            # if the value is int, float, str, or bool, log it
            elif self.is_value(val):
                self.log_data(key, val)
            # if the value is a list or tuple, log it recursively
            elif isinstance(val, (list, tuple)):
                self.log_iterable(val, prefix=key)
            # detect if the value is an image or video
            elif self.is_media(val):
                # handled by media logger
                pass
            elif val is None:
                pass
            # detect if the value is an image
            elif self.is_array(val):
                self.log_data(key, torch.Tensor(val))
            elif self.is_tensor(val):
                self.log_data(key, val)
            # detect if the value is a video
            # elif self.is_video(val):
            #    self.log_data(key, wandb.Video(val))
            # else throw an error
            # else throw an error
            else:
                raise ValueError(f"Cannot log data of type {type(val)}")

    def log_iterable(self, data: Iterable, prefix: Optional[str] = None) -> None:
        """log an iterable to wandb
        Args:
            data (Iterable): data
            prefix (Optional[str], optional): prefix. Defaults to None.

        Raises:
            ValueError: if the value is not of type list or tuple

        Returns:
            None
        """

        assert isinstance(data, (list, tuple)), "data must be a list or tuple"

        for i, val in enumerate(data):
            if prefix is not None:
                key = f"{prefix}/{i}"
            else:
                key = f"{i}"
            # if the value is a dict, log it recursively
            if isinstance(val, dict):
                self.log_dict(val, prefix=key)
            # if the value is int, float, str, or bool, log it
            elif self.is_value(val):
                self.log_data(key, val)
            # if the value is a list or tuple, log it recursively
            elif isinstance(val, (list, tuple)):
                self.log_iterable(val, prefix=key)
            # detect if the value is an image or video
            elif self.is_media(val):
                # handled by media logger
                pass
            elif val is None:
                pass
            # detect if the value is an image
            # elif self.is_figure(val):
            #    self.log_data(key, wandb.Image(val))
            # detect if the value is a video
            # elif self.is_video(val):
            #    self.log_data(key, wandb.Video(val))
            # else throw an error
            else:
                raise ValueError(f"Cannot log data of type {type(val)}")

    def log_data(self, key: str, value) -> None:
        """log data to wandb
        Args:
            key (str): key
            value (int, float, str, bool, torch.Tensor, np.ndarray, wandb.Image, wandb.Video): value

        Raises:
            ValueError: if the value is not of type int, float, str, bool, torch.Tensor, np.ndarray, np.number, wandb.Image, or wandb.Video
        """

        assert isinstance(key, str), "key must be a string"
        assert isinstance(
            value,
            (
                int,
                float,
                str,
                bool,
                torch.Tensor,
                np.ndarray,
                np.number,
                wandb.Image,
                wandb.Video,
            ),
        ), "value must be an int, float, str, bool, torch.Tensor, np.ndarray, np.number, wandb.Image or wandb.Video"
        self.run.log({key: value}, step=self.step_)


class ConsoleLogger(BaseLogger):
    """console logger class"""

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.console = Console()

    def log(
        self, data: dict, prefix: Optional[str] = None, step: Optional[int] = None
    ) -> None:
        """log a dictionary to console
        Args:
            data (dict): data
            prefix (Optional[str], optional): prefix. Defaults to None.
            step (Optional[int], optional): step. Defaults to None.

        Returns:
            None
        """

        if step is not None:
            self.set_step(step)

        self.console.log(f"-" * 20)
        self.console.log(f"Step: {self.step_}")
        if prefix is not None:
            self.console.log(f"Prefix: {prefix}")
        self.console.log(data)


class PandasLogger(BaseLogger):
    """pandas logger class"""

    def __init__(
        self,
        dir: str,
        file: str = "results.csv",
        cfg: Optional[dict] = None,
        mean_arrays: bool = True,
    ) -> None:
        super().__init__()
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = file
        self.path = Path(self.dir, self.file)
        print("results will be saved at", self.path)
        self.reset_local_data()
        self.max_attempts = 60
        self.mean_arrays = mean_arrays
        self.cfg = conf_to_dict(cfg) if cfg is not None else None
        if self.cfg is not None:
            self.log_config()

    def log_data(self, key: str, value) -> None:
        """log data to pandas dataframe
        Args:
            key (str): key
            value (int, float, str, bool): value

        Raises:
            ValueError: if the value is not of type int, float, str, or bool

        Returns:
            None
        """

        assert isinstance(key, str), "key must be a string"
        assert self.is_value(value), "value must be an int, float, str, or bool"
        self.data[key] = value

    def log(
        self, data: dict, prefix: Optional[str] = None, step: Optional[int] = None
    ) -> None:
        """log a dictionary to pandas dataframe
        Args:
            data (dict): data
            prefix (Optional[str], optional): prefix. Defaults to None.
            step (Optional[int], optional): step. Defaults to None.

        Raises:
            ValueError: if the value is not of type dict

        Returns:
            None
        """

        assert isinstance(data, dict), "data must be a dict"

        # check if the step has changed
        if step is not None and self.get_step() < step:
            self.save_to_file()
            self.set_step(step)
            self.reset_local_data()
            self.log_config()

        for key, val in data.items():
            if prefix is not None:
                key = f"{prefix}/{key}"
            # if the value is a dict, log it recursively
            if isinstance(val, dict):
                self.log(val, prefix=key)
            # if the value is int, float, str, or bool, log it
            elif self.is_value(val):
                self.log_data(key, val)
            # if the value is a list or tuple, log it recursively
            elif isinstance(val, (list, tuple)):
                self.log_iterable(val, prefix=key)
            # detect if the value is an image
            elif self.is_media(val):
                # no need to log images
                pass
            # detect if the value is an array
            elif self.is_tensor(val):
                # check if the tensor has only a single value
                if val.numel() == 1:
                    self.log_data(key, val.item())
                else:
                    if self.mean_arrays:
                        self.log_data(key, torch.mean(val).item())
                    else:
                        self.log_iterable(val.tolist(), prefix=key)
            elif self.is_array(val):
                if self.mean_arrays:
                    self.log_data(key, np.mean(val))
                else:
                    self.log_iterable(list(val), prefix=key)
            elif val is None:
                self.log_data(key, "None")
            # else throw an error
            else:
                raise ValueError(f"Cannot log data of type {type(val)}")

    def log_iterable(self, data: Iterable, prefix: Optional[str] = None) -> None:
        """log an iterable to pandas dataframe
        Args:
            data (Iterable): data
            prefix (Optional[str], optional): prefix. Defaults to None.

        Raises:
            ValueError: if the value is not of type list or tuple

        Returns:
            None
        """

        assert isinstance(data, (list, tuple)), "data must be a list or tuple"

        for i, val in enumerate(data):
            if prefix is not None:
                key = f"{prefix}/{i}"
            else:
                key = f"{i}"
            # if the value is a dict, log it recursively
            if isinstance(val, dict):
                self.log(val, prefix=key)
            # if the value is int, float, str, or bool, log it
            elif self.is_value(val):
                self.log_data(key, val)
            # if the value is a list or tuple, log it recursively
            elif isinstance(val, (list, tuple)):
                self.log_iterable(val, prefix=key)
            # detect if the value is an image
            elif self.is_media(val):
                # no need to log images
                pass
            elif val is None:
                self.log_data(key, "None")
            # else throw an error
            else:
                raise ValueError(f"Cannot log data of type {type(val)}")

    def save_to_file(self, path: str = None) -> None:
        """save the dataframe to a file
        Args:
            path (str, optional): path. Defaults to None.

        Returns:
            None
        """

        path = self.path if path is None else Path(path)

        # try to load the file
        print("saving results at", path)
        if path.exists():
            print("updating existing results")
            # try to load the file multiple times because it might be locked
            attempts = 0
            while attempts < self.max_attempts:
                try:
                    df = pd.read_csv(path)
                    break
                except:
                    attempts += 1
                    time.sleep(1.0)
                if attempts == self.max_attempts:
                    print("could not read the file")
                    return
        else:
            df = pd.DataFrame([])

        new_df = pd.DataFrame([self.data])
        # append the dataframe
        df = pd.concat([df, new_df], ignore_index=True)

        # save the dataframe
        attempts = 0
        while attempts < 60:
            try:
                df.to_csv(path, index=False)
                self.result = pd.DataFrame([])
                break
            except:
                attempts += 1
                time.sleep(1.0)
            if attempts == self.max_attempts:
                print("could not save the file")
                return

    def reset_local_data(self) -> None:
        """reset the local dataframe"""
        self.data = {"step": self.get_step()}

    def log_config(self) -> None:
        """log the configuration file"""
        if self.cfg is not None:
            self.log(self.cfg)


class MediaLogger(BaseLogger):
    """media logger class"""

    def __init__(
        self,
        dir: str,
        image_format: str = "png",
        video_format: str = "mp4",
        run: Optional[wandb.run] = None,
    ) -> None:
        super().__init__()
        self.dir = Path(dir, "media")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.image_format = image_format
        self.video_format = video_format
        self.run = run
        print("media will be saved at", self.dir)

    def log_data(self, key: str, value) -> None:
        """log data to pandas dataframe
        Args:
            key (str): key
            value (int, float, str, bool): value

        Raises:
            ValueError: if the value is not of type int, float, str, or bool

        Returns:
            None
        """

        assert isinstance(key, str), "key must be a string"
        assert self.is_media(
            value
        ), "value must be matplotlib figure or numpy array in image shape"

        if self.is_figure(value):
            if isinstance(value, plt.Figure):
                path = Path(self.dir, f"{key} {self.get_step()}.{self.image_format}")
                self.save_fig_to_file(value, path)
                plt.close(value)

            elif isinstance(value, np.ndarray):
                # convert np array to plt figure
                path = Path(self.dir, f"{key} {self.get_step()}.{self.image_format}")
                fig = plt.figure()
                plt.imshow(value)
                plt.axis("off")
                plt.tight_layout()
                self.save_fig_to_file(fig, path)
                plt.close(fig)

            if self.run is not None:
                self.run.log({key: wandb.Image(str(path))}, step=self.get_step())

        elif self.is_video(value):
            if isinstance(value, (animation.FuncAnimation, animation.ArtistAnimation)):
                path = Path(self.dir, f"{key} {self.get_step()}.{self.video_format}")
                self.save_animation_to_file(value, path)

            if self.run is not None:
                self.run.log({key: wandb.Video(str(path))}, step=self.get_step())

    def log(
        self, data: dict, prefix: Optional[str] = None, step: Optional[int] = None
    ) -> None:
        """log a dictionary to pandas dataframe
        Args:
            data (dict): data
            prefix (Optional[str], optional): prefix. Defaults to None.
            step (Optional[int], optional): step. Defaults to None.

        Raises:
            ValueError: if the value is not of type dict

        Returns:
            None
        """

        assert isinstance(data, dict), "data must be a dict"
        assert isinstance(step, (int, None)), "step must be an int or None"

        # check if the step has changed
        if step is not None and self.get_step() < step:
            self.set_step(step)

        for key, val in data.items():
            if prefix is not None:
                key = f"{prefix} {key}"
            # if the value is a dict, log it recursively
            if isinstance(val, dict):
                self.log(val, prefix=key)
            # if the value is int, float, str, or bool, skip it
            elif self.is_value(val):
                pass
            elif self.is_tensor(val):
                pass
            # if the value is a list or tuple, log it recursively
            elif isinstance(val, (list, tuple)):
                self.log_iterable(val, prefix=key)
            # detect if the value is an image
            elif self.is_media(val):
                self.log_data(key, val)
            # else throw an error
            elif self.is_array(val):
                pass
            else:
                raise ValueError(f"Cannot log data of type {type(val)}")

    def log_iterable(self, data: Iterable, prefix: Optional[str] = None) -> None:
        """log an iterable to pandas dataframe
        Args:
            data (Iterable): data
            prefix (Optional[str], optional): prefix. Defaults to None.

        Raises:
            ValueError: if the value is not of type list or tuple

        Returns:
            None
        """

        assert isinstance(data, (list, tuple)), "data must be a list or tuple"

        for i, val in enumerate(data):
            if prefix is not None:
                key = f"{prefix} {i}"
            else:
                key = f"{i}"
            # if the value is a dict, log it recursively
            if isinstance(val, dict):
                self.log(val, prefix=key)
            # if the value is int, float, str, or bool, log it
            elif self.is_value(val):
                pass
            # if the value is a list or tuple, log it recursively
            elif isinstance(val, (list, tuple)):
                self.log_iterable(val, prefix=key)
            # detect if the value is an image
            elif self.is_media(val):
                self.log_data(key, val)
            # else throw an error
            else:
                raise ValueError(f"Cannot log data of type {type(data)}")

    def save_fig_to_file(self, image: plt.Figure, path: Union[str, Path]) -> None:
        """save the dataframe to a file
        Args:
            image (plt.Figure): image
            path (str, Path): path. Defaults to None.

        Returns:
            None
        """

        path = Path(path)

        # try to load the file
        print("saving image at", path)
        if path.exists():
            print("updating existing image!")

        image.savefig(path)

    def save_animation_to_file(
        self, animation: animation, path: Union[str, Path]
    ) -> None:
        """save the dataframe to a file
        Args:
            animation (animation): animation
            path (str, Path): path. Defaults to None.

        Returns:
            None
        """

        path = Path(path)

        # try to load the file
        print("saving animation at", path)
        if path.exists():
            print("updating existing animation!")

        if self.video_format == "mp4":
            animation.save(path)

        elif self.video_format == "gif":
            animation.save(path, writer="imagemagick")

        elif self.video_format == "webm":
            animation.save(path, writer="ffmpeg")

        elif self.video_format == "html":
            with open(path, "w") as f:
                print(animation.to_html5_video(), file=f)
