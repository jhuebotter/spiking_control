import wandb
from .extratyping import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.pretty import pprint
from rich.console import Console
import pandas as pd
from pathlib import Path
import time


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
        assert step >= self.step_, "step must be greater than or equal to the current step"

        self.step_ = step

    def get_step(self) -> int:
        """get the step counter"""
        return self.step_
    
    def log(self, data: dict, prefix: Optional[str] = None, step: Optional[int] = None) -> None:
        """ log a dictionary
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
    
    def is_figure(val) -> bool:
        """detect if the value is a matplotlib figure or an image
        Args:
            val (any): value
        
        Returns:
            bool: True if the value is a matplotlib figure or an numpy array in image shape, else False
        """

        return (isinstance(val, plt.Figure) or \
                isinstance(val, np.ndarray) and len(val.shape) == 3)


class WandBLogger(BaseLogger):
    """wandb logger class"""
    def __init__(
            self,
            run: wandb.run,
        ) -> None:
        
        super().__init__()
        self.run = run
    
    def log(self, data: dict, prefix: Optional[str] = None, step: Optional[int] = None) -> None:
        """ log a dictionary to wandb
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
            elif isinstance(val, (int, float, str, bool)):
                self.log_data(key, val)
            # if the value is a list or tuple, log it recursively
            elif isinstance(val, (list, tuple)):
                self.log_iterable(val, prefix=key)                    
            # detect if the value is an image 
            elif self.is_figure(val):
                self.log_data(key, wandb.Image(val))
            # else throw an error
            else:
                raise ValueError(f"Cannot log data of type {type(data)}")

    def log_iterable(self, data: Iterable, prefix: Optional[str] = None) -> None:
        """ log an iterable to wandb
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
            elif isinstance(val, (int, float, str, bool)):
                self.log_data(key, val)
            # if the value is a list or tuple, log it recursively
            elif isinstance(val, (list, tuple)):
                self.log_iterable(val, prefix=key)
            # detect if the value is an image 
            elif self.is_figure(val):
                self.log_data(key, wandb.Image(val))
            # else throw an error
            else:
                raise ValueError(f"Cannot log data of type {type(data)}")

    def log_data(self, key: str, value) -> None:
        """ log data to wandb
        Args:
            key (str): key
            value (int, float, str, bool, torch.Tensor, np.ndarray, wandb.Image, wandb.Video): value
        
        Raises:
            ValueError: if the value is not of type int, float, str, bool, torch.Tensor, np.ndarray, wandb.Image, or wandb.Video
        """

        assert isinstance(key, str), "key must be a string"
        assert isinstance(value, (int, float, str, bool, torch.Tensor, np.ndarray, wandb.Image, wandb.Video)), \
            "value must be an int, float, str, bool, torch.Tensor, np.ndarray, wandb.Image or wandb.Video"
        self.run.log({key: value}, step=self.step_)


class ConsoleLogger(BaseLogger):
    """console logger class"""
    def __init__(
            self,
        ) -> None:
        
        super().__init__()
        self.console = Console()

    def log(self, data: dict, prefix: Optional[str] = None, step: Optional[int] = None) -> None:
        """ log a dictionary to console
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
        ) -> None:
        
        super().__init__()
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = file
        self.path = Path(self.dir, self.file)
        print("results will be saved at", self.path)
        self.reset_local_data()
        self.max_attempts = 60

    def log_data(self, key: str, value) -> None:
        """ log data to pandas dataframe
        Args:
            key (str): key
            value (int, float, str, bool): value
        
        Raises:
            ValueError: if the value is not of type int, float, str, or bool

        Returns:
            None
        """

        assert isinstance(key, str), "key must be a string"
        assert isinstance(value, (int, float, str, bool)), \
            "value must be an int, float, str, or bool"
        self.data[key] = value

    def log(self, data: dict, prefix: Optional[str] = None, step: Optional[int] = None) -> None:
        """ log a dictionary to pandas dataframe
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
        if step is not None:
            if self.get_step() < step and self.data != {'step': self.get_step()}:
                self.save_to_file()
            self.set_step(step)
            self.reset_local_data()

        for key, val in data.items():
            if prefix is not None:
                key = f"{prefix}/{key}"
            # if the value is a dict, log it recursively
            if isinstance(val, dict):
                self.log(val, prefix=key)
            # if the value is int, float, str, or bool, log it
            elif isinstance(val, (int, float, str, bool)):
                self.log_data(key, val)
            # if the value is a list or tuple, log it recursively
            elif isinstance(val, (list, tuple)):
                self.log_iterable(val, prefix=key)                    
            # detect if the value is an image 
            elif self.is_figure(val):
                # no need to log images
                pass
            # else throw an error
            else:
                raise ValueError(f"Cannot log data of type {type(data)}")

    def log_iterable(self, data: Iterable, prefix: Optional[str] = None) -> None:
        """ log an iterable to pandas dataframe
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
            elif isinstance(val, (int, float, str, bool)):
                self.log_data(key, val)
            # if the value is a list or tuple, log it recursively
            elif isinstance(val, (list, tuple)):
                self.log_iterable(val, prefix=key)
            # detect if the value is an image 
            elif self.is_figure(val):
                # no need to log images
                pass
            # else throw an error
            else:
                raise ValueError(f"Cannot log data of type {type(data)}")

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
        self.data = {'step': self.get_step()}