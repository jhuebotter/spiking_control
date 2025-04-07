from . import BaseAgent
from src.memory import EpisodeMemory, Transition, Episode
from src.models import make_transition_model, make_policy_model
from ..utils import (
    make_optimizer,
    dict_mean,
    FrameStack,
    LinearScheduler,
    ExponentialScheduler,
    StepScheduler,
    LRSchedulerWrapper
)
from src.eval_helpers import baseline_prediction

import gymnasium as gym
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Optional
from pathlib import Path
from copy import deepcopy
from src.plotting import render_video, animate_predictions, TrajectoryPlotter


class PredictiveControlAgent(BaseAgent):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        config: DictConfig,
        device: torch.device,
        loggers: list = [],
        dir: Optional[str] = None,
        eval_env: Optional[gym.vector.VectorEnv] = None,
        id: Optional[str] = None,
    ):

        super().__init__(
            env,
            config,
            device,
            loggers,
            dir,
            eval_env,
            id,
        )

        # initialize config
        self.memory_config = config.agent.memory
        self.transition_config = config.transition
        self.policy_config = config.policy
        self.run_config = config.run
        self.plotting_config = config.plotting

        # training parameters
        self.reset_memory = config.agent.reset_memory
        self.steps_per_iteration = config.agent.steps_per_iteration
        self.steps_per_evaluation = config.agent.steps_per_evaluation
        self.transition_batches_per_iteration = (
            self.transition_config.learning.params.get("batches_per_iteration", 1)
        )
        self.transition_batch_size = self.transition_config.learning.params.get(
            "batch_size", 128
        )
        self.policy_batches_per_iteration = self.policy_config.learning.params.get(
            "batches_per_iteration", 1
        )
        self.policy_batch_size = self.policy_config.learning.params.get(
            "batch_size", 128
        )
        self.eval_first = config.run.get("eval_first", True)

        # initialize dimensions
        self.action_dim = env.unwrapped.action_space.shape[1]
        self.state_dim = env.unwrapped.observation_space["proprio"].shape[1]
        self.target_dim = env.unwrapped.observation_space["target"].shape[1]

        self.setup()

    def setup(self):

        # initialize memory
        self.memory = EpisodeMemory(**self.memory_config.params)

        # initialize models
        self.transition_model = make_transition_model(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            config=self.transition_config.model,
        ).to(self.device)

        self.policy_model = make_policy_model(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            target_dim=self.target_dim,
            config=self.policy_config.model,
        ).to(self.device)
        self.models = [self.transition_model, self.policy_model]

        # initialize optimizers
        self.transition_model.set_optimizer(
            make_optimizer(
                self.transition_model, self.transition_config.get("optimizer", {})
            )
        )

        self.policy_model.set_optimizer(
            make_optimizer(self.policy_model, self.policy_config.get("optimizer", {}))
        )

        # make a learning rate scheduler
        transition_model_lr_scheduler = ExponentialScheduler(
            start=self.transition_config.optimizer.params.get("lr", 1e-3),
            end=self.run_config.get("lr_end", 0),
            gamma=self.run_config.get("lr_decay", 1.0),
            warmup_steps=self.run_config.get("lr_warmup_steps", 0),
        )
        self.transition_model_lr_scheduler = LRSchedulerWrapper(
            self.transition_model.optimizer, transition_model_lr_scheduler
        )

        policy_model_lr_scheduler = ExponentialScheduler(
            start=self.policy_config.optimizer.params.get("lr", 1e-3),
            end=self.run_config.get("lr_end", 0),
            gamma=self.run_config.get("lr_decay", 1.0),
            warmup_steps=self.run_config.get("lr_warmup_steps", 0),
        )
        self.policy_model_lr_scheduler = LRSchedulerWrapper(
            self.policy_model.optimizer, policy_model_lr_scheduler
        )

        # make a teacher forcing scheduler
        self.transition_model_tf_scheduler = LinearScheduler(
            self.run_config.get("teacher_forcing_start", 1.0),
            end=self.run_config.get("teacher_forcing_end", 1.0),
            warmup_steps=self.run_config.get("teacher_forcing_warmup_steps", 0),
            decay_steps=self.run_config.get("teacher_forcing_decay_steps", 0),
        )

        # make a noise scheduler
        self.policy_model_noise_scheduler = ExponentialScheduler(
            self.run_config.get("action_noise_std", 0.0),
            gamma=self.run_config.get("action_noise_decay", 1.0),
        )

        # make a action reg weight scheduler
        self.policy_model_action_reg_weight_scheduler = StepScheduler(
            start=self.run_config.get("action_reg_weight_start", 0.0),
            end=self.run_config.get("action_reg_weight_end", 0.0),
            warmup_steps=self.run_config.get("action_reg_weight_warmup_steps", 0),
        )
        self.policy_model_action_smoothness_reg_weight_scheduler = StepScheduler(
            start=self.run_config.get("action_smoothness_reg_weight_start", 0.0),
            end=self.run_config.get("action_smoothness_reg_weight_end", 0.0),
            warmup_steps=self.run_config.get(
                "action_smoothness_reg_weight_warmup_steps", 0
            ),
        )

        # wrap the environment with a video recorder if needed
        if hasattr(self.env.unwrapped, "manual_video"):
            self.manual_video = self.env.unwrapped.manual_video
        else:
            self.manual_video = True
        if not self.manual_video:
            trigger = lambda x: False
            self.env = gym.wrappers.RecordVideo(
                self.env,
                video_folder=Path(self.dir, "media"),
                episode_trigger=trigger,
                name_prefix="recording",
            )

        # initialize counters
        self.steps = 0
        self.episodes = 0
        self.epochs = 0
        self.iterations = 0
        self.policy_updates = 0
        self.transition_updates = 0

        # initialize early stopping
        self.early_stop_metric = self.run_config.get("early_stop_metric", None)
        self.early_stop_mode = self.run_config.get("early_stop_mode", "min").lower()
        self.early_stop_patience = self.run_config.get("early_stop_patience", 0)
        self.early_stop_metric_list = []

    def run(self, total_steps: int):

        if self.eval_first:
            self.test(steps=self.steps_per_evaluation, render=True)
        while self.steps < total_steps:
            self.collect_rollouts(self.steps_per_iteration, self.reset_memory)
            self.train()
            render = (
                (self.iterations + 1) % self.plotting_config.get("render_every", 1)
            ) == 0
            test_results = self.test(
                steps=self.steps_per_evaluation,
                render=render,
                return_results=[self.early_stop_metric],
            )
            if self.run_config.get("save_latest_model", False):
                self.save_models()
            self.step_scheduler()
            self.iterations += 1
            if self.check_early_stop(test_results):
                break

        self.finish_run()

    def step_scheduler(self):

        # first get the current parameters from the schedulers
        transition_lr = self.transition_model_lr_scheduler.get_value()
        policy_lr = self.policy_model_lr_scheduler.get_value()
        transition_teacher_forcing_p = self.transition_model_tf_scheduler.get_value()
        policy_noise_std = self.policy_model_noise_scheduler.get_value()
        action_reg_weight = self.policy_model_action_reg_weight_scheduler.get_value()
        action_smoothness_reg_weight = (
            self.policy_model_action_smoothness_reg_weight_scheduler.get_value()
        )
        # log them
        self.log(
            {
                "transition model learning rate": transition_lr,
                "policy model learning rate": policy_lr,
                "transition model teacher forcing p": transition_teacher_forcing_p,
                "policy model noise std": policy_noise_std,
                "policy model action reg weight": action_reg_weight,
                "policy model action smoothness reg weight": action_smoothness_reg_weight,
            },
            step=self.epochs,
        )
        # step the schedulers
        self.transition_model_lr_scheduler.step()
        self.policy_model_lr_scheduler.step()
        self.transition_model_tf_scheduler.step()
        self.policy_model_noise_scheduler.step()
        self.policy_model_action_reg_weight_scheduler.step()
        self.policy_model_action_smoothness_reg_weight_scheduler.step()

    def check_early_stop(self, test_results):

        patience = self.early_stop_patience
        metric = self.early_stop_metric
        mode = self.early_stop_mode
        assert mode in ["min", "max"], "early stopping mode must be 'min' or 'max'"

        if patience > 0 and metric is not None:
            print(
                f"Iteration {self.iterations}: Checking early stopping with patience {patience} in mode {mode} for metric {metric}"
            )
            if metric in test_results.keys():
                print("current value: ", test_results[metric])
                self.early_stop_metric_list.append(test_results[metric])
                if mode == "min":
                    best = min(self.early_stop_metric_list)
                elif mode == "max":
                    best = max(self.early_stop_metric_list)
                print(
                    "best value: ",
                    best,
                    "in iteration: ",
                    self.early_stop_metric_list.index(best),
                )
                # check if the best value has not been improved for patience iterations
                last_values = self.early_stop_metric_list[-patience:]
                if best not in last_values:
                    print(
                        f"Early stopping triggered after {self.iterations} iterations"
                    )
                    return True
            else:
                print(f"metric {metric} not found in test results")
                return False
        return False

    @torch.no_grad()
    def collect_rollouts(self, steps: int, reset_memory: bool = True):

        self.policy_model.eval()

        if reset_memory:
            self.memory.reset()

        set_test_mode_fn = getattr(self.env.unwrapped, "set_test_mode", None)
        if callable(set_test_mode_fn):
            print("setting test mode to False")
            set_test_mode_fn(False)

        num_envs = self.env.unwrapped.num_envs

        episodes = [Episode() for _ in range(num_envs)]

        action_min = torch.tensor(
            self.env.unwrapped.action_space.low, device=self.device
        )
        action_max = torch.tensor(
            self.env.unwrapped.action_space.high, device=self.device
        )

        # reset the environment
        observations, infos = self.env.reset()
        # check if the observation dict contains tensors or arrays
        if isinstance(observations["proprio"], torch.Tensor):
            obs = observations["proprio"].clone()
            targets = observations["target"].clone()
        else:
            obs = torch.tensor(
                observations["proprio"], device=self.device, dtype=torch.float32
            )
            targets = torch.tensor(
                observations["target"], device=self.device, dtype=torch.float32
            )

        self.policy_model.reset_state()

        step = 0
        total_reward = 0
        completed_episodes = 0
        success_tracker = []
        steps_to_target_tracker = []
        steps_on_target_tracker = []
        cumulative_distance_tracker = []
        with tqdm(
            total=steps,
            desc=f"{'obtaining experience':30}",
        ) as pbar:
            while step < steps:

                # predict the action
                actions = self.policy_model.predict(obs, targets)
                actions = actions.squeeze(0).clamp(action_min, action_max).detach()

                # Add Gaussian noise
                noise_std = self.policy_model_noise_scheduler.get_value()
                if noise_std > 0:
                    noise = (
                        torch.randn_like(actions) * noise_std
                    )  # Generate Gaussian noise
                    actions = (actions + noise).clamp(
                        action_min, action_max
                    )  # Apply noise and re-clamp

                # step the environment
                observations, rewards, terminates, truncateds, infos = self.env.step(
                    actions
                )

                dones = [
                    True if ter or tru else False
                    for ter, tru in zip(terminates, truncateds)
                ]
                if isinstance(observations["proprio"], torch.Tensor):
                    next_obs = observations["proprio"].clone()
                else:
                    next_obs = torch.tensor(
                        observations["proprio"],
                        device=self.device,
                        dtype=torch.float32,
                    )

                # store the transition
                for i, (o, t, a, r, d, no) in enumerate(
                    zip(obs, targets, actions, rewards, dones, next_obs)
                ):
                    # ! This is a hack
                    if not d:
                        episodes[i].append(Transition(o, t, a, r, d, no))
                    if d:
                        # ! This only adds complete episodes to the memory
                        self.memory.append(episodes[i])
                        total_reward += episodes[i].get_cumulative_reward()
                        episodes[i] = Episode()
                        completed_episodes += 1
                        self.episodes += 1
                        if "success" in infos:
                            success_tracker.append(1 if infos["success"][i] else 0)
                        if "steps_to_target" in infos:
                            steps_to_target_tracker.append(infos["steps_to_target"][i])
                        if "steps_on_target" in infos:
                            steps_on_target_tracker.append(infos["steps_on_target"][i])
                        if "cumulative_distance" in infos:
                            cumulative_distance_tracker.append(
                                infos["cumulative_distance"][i]
                            )

                # update the state
                if isinstance(observations["proprio"], torch.Tensor):
                    obs = observations["proprio"].clone()
                    targets = observations["target"].clone()
                else:
                    obs = torch.tensor(
                        observations["proprio"],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    targets = torch.tensor(
                        observations["target"],
                        device=self.device,
                        dtype=torch.float32,
                    )

                step += num_envs
                pbar.update(num_envs)

        self.steps += step
        average_reward = total_reward / completed_episodes
        if len(success_tracker) > 0:
            average_success_rate = sum(success_tracker) / len(success_tracker)
        else:
            average_success_rate = None
        if len(steps_to_target_tracker) > 0:
            average_steps_to_target = sum(steps_to_target_tracker) / len(
                steps_to_target_tracker
            )
        else:
            average_steps_to_target = None
        if len(cumulative_distance_tracker) > 0:
            average_cumulative_distance = sum(cumulative_distance_tracker) / len(
                cumulative_distance_tracker
            )
        else:
            average_cumulative_distance = None
        if len(steps_on_target_tracker) > 0:
            average_steps_on_target = sum(steps_on_target_tracker) / len(
                steps_on_target_tracker
            )
        results = {
            "train average reward": average_reward,
            "train average success rate": average_success_rate,
            "train average steps to target": average_steps_to_target,
            "train average steps on target": average_steps_on_target,
            "train average cumulative distance": average_cumulative_distance,
        }

        self.log(results, step=self.epochs)

    def train_transition_model(self):

        # train the transition model
        transition_results = []
        n_transition_batches = self.transition_config.learning.get(
            "batches_per_iteration", 1
        )
        teacher_forcing_p = self.transition_model_tf_scheduler.get_value()
        pbar = tqdm(
            range(n_transition_batches), desc=f"{'training transition model':30}"
        )
        for batch in pbar:
            transition_result = self.transition_model.train_fn(
                memory=self.memory,
                record=False,  # True,
                excluded_monitor_keys=self.transition_model.plot_monitors
                + self.transition_model.numeric_monitors,
                teacher_forcing_p=teacher_forcing_p,
                **self.transition_config.get("learning", {}).get("params", {}),
            )
            self.transition_updates += 1
            loss = transition_result["loss"]
            pbar.set_postfix_str(f"loss: {loss:.5f}")
            transition_results.append(transition_result)

        return transition_results

    def train_policy_model(self):

        # train the policy model
        action_reg_weight = self.policy_model_action_reg_weight_scheduler.get_value()
        action_smoothness_reg_weight = (
            self.policy_model_action_smoothness_reg_weight_scheduler.get_value()
        )
        policy_results = []
        n_policy_batches = self.policy_config.learning.get("batches_per_iteration", 1)
        pbar = tqdm(range(n_policy_batches), desc=f"{'training policy model':30}")
        for batch in pbar:
            policy_result = self.policy_model.train_fn(
                memory=self.memory,
                transition_model=self.transition_model,
                loss_gain=self.env.unwrapped.call("get_loss_gain")[0],
                action_reg_weight=action_reg_weight,
                action_smoothness_reg_weight=action_smoothness_reg_weight,
                record=False,  # True,
                excluded_monitor_keys=self.policy_model.plot_monitors
                + self.policy_model.numeric_monitors,
                **self.policy_config.get("learning", {}).get("params", {}),
            )
            self.policy_updates += 1
            loss = policy_result["loss"]
            pbar.set_postfix_str(f"loss: {loss:.5f}")
            policy_results.append(policy_result)

        return policy_results

    def train_epoch(self):

        # train the models
        transition_results = self.train_transition_model()
        policy_results = self.train_policy_model()

        return transition_results, policy_results

    def train(self, epochs: int = 1):

        for e in range(epochs):
            transition_results, policy_results = self.train_epoch()

            self.epochs += 1

            # collect the training results
            results = {
                "steps": self.steps,
                "episodes": self.episodes,
                "epochs": self.epochs,
                "policy model updates": self.policy_updates,
                "transition model updates": self.transition_updates,
                "policy model parameters": self.policy_model.count_parameters(),
                "transition model parameters": self.transition_model.count_parameters(),
            }
            results.update(
                dict_mean(transition_results, prefix=self.transition_model.name + " ")
            )
            results.update(
                dict_mean(policy_results, prefix=self.policy_model.name + " ")
            )

            # log the results
            self.log(results, step=self.epochs)

    @torch.no_grad()
    def test(
        self,
        steps: int,
        env: Optional[gym.vector.VectorEnv] = None,
        render: bool = False,
        return_results: list = [],
    ):

        if env is None:
            env = self.env if self.eval_env is None else self.eval_env

        set_test_mode_fn = getattr(env.unwrapped, "set_test_mode", None)
        if callable(set_test_mode_fn):
            print("setting test mode to True")
            set_test_mode_fn(True)

        num_envs = env.unwrapped.num_envs

        action_min = torch.tensor(env.unwrapped.action_space.low, device=self.device)
        action_max = torch.tensor(env.unwrapped.action_space.high, device=self.device)

        observations, infos = env.reset()
        if isinstance(observations["proprio"], torch.Tensor):
            obs = observations["proprio"].clone().detach()
            targets = observations["target"].clone().detach()
        else:
            obs = torch.tensor(
                observations["proprio"], device=self.device, dtype=torch.float32
            )
            targets = torch.tensor(
                observations["target"], device=self.device, dtype=torch.float32
            )

        completed_episodes = []
        episodes = [Episode() for _ in range(num_envs)]

        success_tracker = []
        steps_to_target_tracker = []
        steps_on_target_tracker = []
        cumulative_distance_tracker = []

        if render:
            # check if the environment is wrapped with RecordVideo
            if not self.manual_video:
                env.start_recording(f"recording_{self.iterations+1}")
            else:
                completed_framestacks = []
                framestacks = [FrameStack() for _ in range(num_envs)]
                frames = env.call("render", mode="rgb_array")
                for i, frame in enumerate(frames):
                    framestacks[i].append(frame)

        with torch.no_grad():
            self.policy_model.eval()
            self.policy_model.reset_state()
            self.transition_model.eval()
            self.transition_model.reset_state()

            step = 0
            total_reward = 0
            with tqdm(
                total=steps,
                desc=f"{'evaluating models':30}",
            ) as pbar:
                while step < steps:

                    # predict the action
                    actions = self.policy_model.predict(obs, targets, record=True)
                    actions = actions.squeeze(0).clamp(action_min, action_max).detach()

                    # predict the next states
                    _ = self.transition_model.predict(
                        obs, actions, deterministic=True, record=True
                    )

                    # step the environment
                    observations, rewards, terminates, truncateds, infos = env.step(
                        actions
                    )

                    dones = [
                        True if ter or tru else False
                        for ter, tru in zip(terminates, truncateds)
                    ]

                    if isinstance(observations["proprio"], torch.Tensor):
                        next_obs = observations["proprio"].clone().detach()
                    else:
                        next_obs = torch.tensor(
                            observations["proprio"],
                            device=self.device,
                            dtype=torch.float32,
                        )

                    # store the transition
                    for i, (o, t, a, r, d, no) in enumerate(
                        zip(obs, targets, actions, rewards, dones, next_obs)
                    ):
                        # ! This is a hack
                        if not d:
                            episodes[i].append(Transition(o, t, a, r, d, no))
                        if d:
                            completed_episodes.append(episodes[i])
                            total_reward += episodes[i].get_cumulative_reward()
                            episodes[i] = Episode()
                            if "success" in infos:
                                success_tracker.append(1 if infos["success"][i] else 0)
                            if "steps_to_target" in infos:
                                steps_to_target_tracker.append(
                                    infos["steps_to_target"][i]
                                )
                            if "steps_on_target" in infos:
                                steps_on_target_tracker.append(
                                    infos["steps_on_target"][i]
                                )
                            if "cumulative_distance" in infos:
                                cumulative_distance_tracker.append(
                                    infos["cumulative_distance"][i]
                                )

                    # store frames for rendering
                    if render and self.manual_video:
                        frames = env.call("render", mode="rgb_array")
                        for i, frame in enumerate(frames):
                            framestacks[i].append(frame)
                            if dones[i]:
                                completed_framestacks.append(framestacks[i])
                                framestacks[i] = FrameStack()

                    # update the state
                    if isinstance(observations["proprio"], torch.Tensor):
                        obs = observations["proprio"].clone()
                        targets = observations["target"].clone()
                    else:
                        obs = torch.tensor(
                            observations["proprio"],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        targets = torch.tensor(
                            observations["target"],
                            device=self.device,
                            dtype=torch.float32,
                        )

                    step += num_envs
                    pbar.update(num_envs)

            if not self.manual_video and render:
                env.stop_recording()
            
        # log the results
        average_reward = total_reward / len(completed_episodes)
        if len(success_tracker) > 0:
            average_success_rate = sum(success_tracker) / len(success_tracker)
        else:
            average_success_rate = None
        if len(steps_to_target_tracker) > 0:
            average_steps_to_target = sum(steps_to_target_tracker) / len(
                steps_to_target_tracker
            )
        else:
            average_steps_to_target = None
        if len(cumulative_distance_tracker) > 0:
            average_cumulative_distance = sum(cumulative_distance_tracker) / len(
                cumulative_distance_tracker
            )
        else:
            average_cumulative_distance = None
        if len(steps_on_target_tracker) > 0:
            average_steps_on_target = sum(steps_on_target_tracker) / len(
                steps_on_target_tracker
            )
        results = {
            "test average reward": average_reward,
            "test average success rate": average_success_rate,
            "test average steps to target": average_steps_to_target,
            "test average steps on target": average_steps_on_target,
            "test average cumulative distance": average_cumulative_distance,
        }

        # get some more numeric montior results
        policy_numeric_monitor_results = self.policy_model.get_monitor_data(
            exclude=self.policy_model.plot_monitors
        )
        transition_numeric_monitor_results = self.transition_model.get_monitor_data(
            exclude=self.transition_model.plot_monitors
        )
        results.update(policy_numeric_monitor_results)
        results.update(transition_numeric_monitor_results)

        # make the video
        if render:
            trajectory_plotter = TrajectoryPlotter(env)
            figures, animations = trajectory_plotter(completed_episodes)
            trajectory_plots = {
                "test trajectory plots": figures,
                "test trajectory animations": animations,
            }
            policy_plots = self.policy_model.get_monitor_data(
                exclude=self.policy_model.numeric_monitors
            )
            transition_plots = self.transition_model.get_monitor_data(
                exclude=self.transition_model.numeric_monitors
            )
            prediction_videos = {
                "test prediction animations": animate_predictions(
                    completed_episodes,
                    self.transition_model,
                    labels=env.unwrapped.get_attr("state_labels")[0],
                    warmup=self.transition_config.learning.params.get(
                        "warmup_steps", 0
                    ),
                    unroll=self.plotting_config.get("prediction_animation_unroll", 1),
                    step=self.plotting_config.get("prediction_animation_step", 2),
                )
            }
            results.update(prediction_videos)
            results.update(policy_plots)
            results.update(transition_plots)
            results.update(trajectory_plots)
            if self.manual_video:
                episode_videos = {"test episodes": render_video(completed_framestacks)}
                results.update(episode_videos)

        prediction_results = baseline_prediction(
            transition_model=self.transition_model,
            episodes=completed_episodes,
            warmup=self.transition_config.learning.params.get("warmup_steps", 0),
            unroll=self.run_config.get("prediction_unroll", 1),
        )
        results.update(prediction_results)
        return_dict = {k: deepcopy(results[k]) for k in return_results}
        self.log(results, step=self.epochs)

        return return_dict

    def save_models(self):

        self.save_transition_model()
        self.save_policy_model()

    def save_transition_model(
        self,
        dir: Optional[str] = None,
        file: str = "transition_model.cpt",
        save_optimizer: bool = True,
    ):

        if dir is None:
            dir = Path(self.dir, "models")

        self.save(
            model=self.transition_model,
            dir=dir,
            file=file,
            optimizer=self.transition_model.get_optimizer() if save_optimizer else None,
        )

    def save_policy_model(
        self,
        dir: Optional[str] = None,
        file: str = "policy_model.cpt",
        save_optimizer: bool = True,
    ):

        if dir is None:
            dir = Path(self.dir, "models")

        self.save(
            model=self.policy_model,
            dir=dir,
            file=file,
            optimizer=self.policy_model.get_optimizer() if save_optimizer else None,
        )

    def load_models(self, dir: Optional[str] = None):

        self.load_transition_model(dir)
        self.load_policy_model(dir)

    def load_transition_model(
        self, dir: Optional[str] = None, file: str = "transition_model.cpt"
    ):

        if dir is None:
            dir = self.run_config.get("load_path", None)
        assert dir is not None, "No load path specified!"
        dir = Path(dir)
        path = Path(dir, file)

        self.transition_model, op = self.load(
            model=self.transition_model,
            path=path,
            optim=self.transition_model.get_optimizer(),
        )
        if op is not None:
            self.transition_model.set_optimizer(op)

    def load_policy_model(
        self, dir: Optional[str] = None, file: str = "policy_model.cpt"
    ):

        if dir is None:
            dir = self.run_config.get("load_path", None)
            assert dir is not None, "No load path specified!"
            dir = Path(dir)
        path = Path(dir, file)

        self.policy_model, op = self.load(
            model=self.policy_model, path=path, optim=self.policy_model.get_optimizer()
        )
        if op is not None:
            self.policy_model.set_optimizer(op)
