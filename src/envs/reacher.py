import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import pygame
from pygame import gfxdraw
import numpy as np
from typing import Optional
from copy import deepcopy


WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)


class ReacherEnv(gym.Env):
    def __init__(
        self,
        seed: int = None,
        max_episode_steps: int = 200,
        render_mode: str = "human",
        moving_target: float = 0.0,
        max_acceleration: float = 10.0,
        fully_observable: bool = True,
        show_target_arm: bool = False,
        include_velocity: bool = True,
        delta_in_obs: bool = False,
        dt: float = 0.02,
        eval: bool = False,
        full_target_obs: bool = False,
        observation_noise_std: float = 0.01,
        zoom: float = 1.0,
        control: str = "velocity",
        **kwargs
    ):
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(1.0 / dt),
        }

        self.dt = dt  # seconds between state updates
        self.max_episode_steps = max_episode_steps
        self.min_action = -1.0
        self.max_action = 1.0
        self.max_acceleration = max_acceleration
        self.damp = 5.0
        self.control = control.lower()
        assert self.control in ["velocity", "acceleration"]

        #####

        self.full_target_obs = full_target_obs
        self.target_dim = 8 if self.full_target_obs else 2
        self.include_velocity = include_velocity
        self.delta_in_obs = delta_in_obs

        #####

        self.eval = eval

        self.max_vel = np.pi / 2.0
        self.min_vel = -self.max_vel

        self.random_target = True
        self.moving_target = moving_target
        self.done_on_target = False
        self.epsilon = 0.05

        self.l1 = 0.5
        self.l2 = 0.4
        self.max_reach = self.l1 + self.l2

        self.render_mode = render_mode
        self.show_target_arm = show_target_arm
        self.screen_px = int(256 * zoom)

        n_noise = 8 if self.include_velocity else 6
        if self.delta_in_obs:
            n_obs = 2 * n_noise
        else:
            n_obs = n_noise

        self.process_noise_std = np.array([0.0, 0.0, 0.0, 0.0])
        self.observation_noise_std = np.ones(n_noise) * observation_noise_std

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(2,)
        )

        self.fully_observable = fully_observable
        if self.fully_observable:
            self.observation_space = spaces.Dict(
                {
                    "proprio": spaces.Box(
                        low=np.array([-1.0] * n_obs), high=np.array([1.0] * n_obs)
                    ),
                    "target": spaces.Box(
                        low=np.array([-1.0] * self.target_dim),
                        high=np.array([1.0] * self.target_dim),
                    ),
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.screen_px, self.screen_px, 3),
                dtype=np.float32,
            )

        self.state_labels = [
            "hand x",
            "hand y",
            "sin alpha",
            "cos alpha",
            "sin beta",
            "cos beta",
        ]
        if self.include_velocity:
            self.state_labels += [
                "vel alpha",
                "vel beta",
            ]

        if self.delta_in_obs:
            self.state_labels += ["delta " + l for l in self.state_labels]

        self.target_labels = [
            "hand x",
            "hand y",
        ]

        self.loss_gain = {"gain": np.array([1.0, 1.0]), "use": np.array([0, 1])}

        self.set_seed(seed)
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state = None
        self.last_state = None
        self.target = None

        self.manual_video = True

    def set_seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)
        return [self.seed]

    def get_seed(self):
        return self.seed

    def get_loss_gain(self):
        return self.loss_gain

    def stepPhysics(self, action):
        angles = self.state[:2]  # get last joint angles
        vel = self.state[2:]  # get last joint velocity

        # get change in state since last update
        dadt = vel
        angles += dadt * self.dt

        if self.control == "acceleration":
            dvdt = action * self.max_acceleration - self.damp * vel
            vel += dvdt * self.dt
        elif self.control == "velocity":
            vel = action * self.max_vel

        # clip velocity in allowed limits
        vel = np.clip(vel, self.min_vel, self.max_vel)

        # avoid very small velocity residues that bounce due to dampening
        vel[np.abs(vel) < 0.01 * self.max_vel] = 0.0

        return np.hstack([angles, vel])

    def stepTarget(self):
        angles = self.target[:2]  # get last joint angles
        vel = self.target[2:]  # get last joint velocity

        # get change in state since last update
        dadt = vel
        dvdt = 0.0
        # update target state
        angles += dadt * self.dt
        vel += dvdt * self.dt

        # clip velocity in allowed limits
        vel = np.clip(vel, self.min_vel, self.max_vel)

        # avoid very small velocity residues that bounce due to dampening
        vel[np.abs(vel) < 0.01 * self.max_vel] = 0.0

        self.target = np.hstack([angles, vel])

    def step(self, action):
        # check if action is a tensor and convert to numpy array
        if hasattr(action, "numpy"):
            action = action.cpu().numpy()
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        # update state
        self.last_state = deepcopy(self.state)
        self.state = np.array(self.stepPhysics(action))
        self.state = self.np_random.normal(self.state, self.process_noise_std)
        self.stepTarget()
        self.episode_step_count += 1

        # make observation
        target_observation = self.make_observation(self.target, noise=False)

        # calculate reward
        ob_rew = self.make_observation(self.state, noise=False)
        reward = -np.linalg.norm(ob_rew[:2] - target_observation[:2]) * self.dt

        if self.fully_observable:
            proprio_observation = self.make_observation(self.state)
            if self.delta_in_obs:
                delta_observation = (
                    ob_rew - self.make_observation(self.last_state, noise=False)
                ) / (self.dt * self.max_vel)
                delta_observation = self.np_random.normal(
                    delta_observation, self.observation_noise_std
                )
                total_observation = np.hstack([proprio_observation, delta_observation])
            else:
                total_observation = proprio_observation
            observation = {
                "proprio": total_observation,
                "target": target_observation[: self.target_dim],
            }
        else:
            observation = self.render(mode="rgb_array")

        # check if episode is done
        terminated = False
        truncated = False
        on_target = False
        if np.allclose(ob_rew[:2], target_observation[:2], atol=self.epsilon):
            on_target = True
            self.steps_on_target += 1
            if self.steps_to_target == self.max_episode_steps:
                self.steps_to_target = self.episode_step_count
            self.was_on_target = True
            if self.done_on_target:
                terminated = True
        max_steps = False
        if self.max_episode_steps and self.episode_step_count == self.max_episode_steps:
            truncated = True
            max_steps = True

        # calculate euclidean distance to target ee position
        dist = np.linalg.norm(ob_rew[:2] - target_observation[:2])
        self.cumulative_distance += dist * self.dt

        # additional info
        info = {
            "on_target": on_target,
            "success": self.was_on_target,
            "steps_to_target": self.steps_to_target,
            "steps_on_target": self.steps_on_target,
            "max_steps": max_steps,
            "step": self.episode_step_count,
            "cumulative_distance": self.cumulative_distance,
        }

        return observation, reward, terminated, truncated, info

    def make_observation(self, state, noise=True):
        a1 = state[0]
        a2 = state[1]

        p1_x = self.l1 * np.cos(a1)
        p1_y = self.l1 * np.sin(a1)

        hand_x = (p1_x + self.l2 * np.cos(a1 + a2)) / self.max_reach
        hand_y = (p1_y + self.l2 * np.sin(a1 + a2)) / self.max_reach

        norm_vel1 = state[2] / self.max_vel
        norm_vel2 = state[3] / self.max_vel

        observation = np.array(
            [hand_x, hand_y, np.sin(a1), np.cos(a1), np.sin(a2), np.cos(a2)]
        )

        if self.include_velocity:
            observation = np.hstack(
                [
                    observation,
                    np.array([norm_vel1, norm_vel2]),
                ]
            )

        if noise:
            observation = self.np_random.normal(observation, self.observation_noise_std)

        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.episode_step_count = 0
        self.steps_on_target = 0
        self.cumulative_distance = 0.0

        state = None
        target = None
        if options is not None:
            state = options.get("state", None)
            target = options.get("target", None)

        if seed is not None:
            self.set_seed(seed)

        # if in eval mode, make a deterministic environment
        if self.eval:
            if self.seed is None:
                self.set_seed(1)
            v = self.seed % 8
            state = np.zeros(4)
            target = np.zeros(4)
            target[:2] = 2 * np.pi * v / 8.0
            if self.moving_target:
                target[2:] = self.max_vel * np.array([0.0, 0.5])

        # state is [theta_1, theta_2, \dot{theta_1}, \dot{theta_2}]
        if state is None:
            self.state = np.zeros(4)
            self.state[:2] = self.np_random.uniform(low=0.0, high=2 * np.pi, size=(2,))
        else:
            self.state = state
        self.last_state = self.state

        if target is None:
            self.target = np.zeros(4)
            if self.random_target:
                self.target[:2] = self.np_random.uniform(
                    low=0.0, high=2 * np.pi, size=(2,)
                )
                if self.np_random.uniform() < self.moving_target:
                    self.target[2:] = self.np_random.uniform(
                        low=0.3 * self.min_vel, high=0.3 * self.max_vel, size=(2,)
                    )
            else:
                self.target[:2] = np.pi, np.pi

        else:
            self.target = target

        # make observation
        target_observation = self.make_observation(self.target, noise=False)
        if self.fully_observable:
            proprio_observation = self.make_observation(self.state)
            if self.delta_in_obs:
                delta_observation = proprio_observation - self.make_observation(
                    self.last_state
                )
                total_observation = np.hstack([proprio_observation, delta_observation])
            else:
                total_observation = proprio_observation
            observation = {
                "proprio": total_observation,
                "target": target_observation[: self.target_dim],
            }
        else:
            observation = self.render(mode="rgb_array")

        # additional info
        on_target = False
        ob_rew = self.make_observation(self.state, noise=False)
        if np.allclose(ob_rew[:2], target_observation[:2], atol=self.epsilon):
            on_target = True
        self.was_on_target = on_target
        self.steps_to_target = self.max_episode_steps if not on_target else 0
        max_steps = False

        info = {
            "on_target": on_target,
            "success": self.was_on_target,
            "steps_to_target": self.steps_to_target,
            "steps_on_target": self.steps_on_target,
            "max_steps": max_steps,
            "step": self.episode_step_count,
            "cumulative_distance": self.cumulative_distance,
        }

        return observation, info

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode

        screen_width, screen_height = self.screen_px, self.screen_px

        center_x = screen_width / 2
        center_y = screen_height / 2

        a1 = self.state[0]
        a2 = self.state[1]

        p1_x = self.l1 * np.cos(a1) * center_x + center_x
        p1_y = self.l1 * np.sin(a1) * center_y + center_y

        p2_x = p1_x + self.l2 * np.cos(a1 + a2) * center_x
        p2_y = p1_y + self.l2 * np.sin(a1 + a2) * center_y

        at1 = self.target[0]
        at2 = self.target[1]

        t1_x = self.l1 * np.cos(at1) * center_x + center_x
        t1_y = self.l1 * np.sin(at1) * center_y + center_y

        t2_x = t1_x + self.l2 * np.cos(at1 + at2) * center_x
        t2_y = t1_y + self.l2 * np.sin(at1 + at2) * center_y

        # prepare screen
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill(WHITE)

        # draw the target
        rad = int(screen_width / 50)
        if self.show_target_arm:
            pygame.draw.line(
                self.surf,
                GRAY,
                (center_x, center_y),
                (t1_x, t1_y),
                int(np.rint(1.2 * rad)),
            )
            pygame.draw.line(self.surf, GRAY, (t1_x, t1_y), (t2_x, t2_y), rad)

            gfxdraw.filled_circle(
                self.surf, int(np.rint(t1_x)), int(np.rint(t1_y)), rad, GRAY
            )

        gfxdraw.filled_circle(
            self.surf, int(np.rint(t2_x)), int(np.rint(t2_y)), rad, BLUE
        )

        # draw the arm
        pygame.draw.line(
            self.surf,
            BLACK,
            (center_x, center_y),
            (p1_x, p1_y),
            int(np.rint(1.2 * rad)),
        )
        pygame.draw.line(self.surf, BLACK, (p1_x, p1_y), (p2_x, p2_y), rad)

        gfxdraw.filled_circle(
            self.surf,
            int(np.rint(center_x)),
            int(np.rint(center_y)),
            int(np.rint(1.2 * rad)),
            BLACK,
        )

        gfxdraw.filled_circle(
            self.surf, int(np.rint(p1_x)), int(np.rint(p1_y)), rad, BLACK
        )

        gfxdraw.filled_circle(
            self.surf, int(np.rint(p2_x)), int(np.rint(p2_y)), rad, RED
        )

        # update the screen
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        if mode == "rgb_array":
            return (
                np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
                )
                / 255.0
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == "__main__":

    env = ReacherEnv(seed=4, show_target_arm=True, fully_observable=False)
    target = np.array([0.75, 0.75, 0.2, 0.2])
    target[:2] *= 2 * np.pi
    target[2:] *= env.max_vel
    observation, info = env.reset(options={"target": target})
    print(observation)
    print(info)
    env.render()
    for i in range(200):
        a = env.action_space.sample()
        if i < 100:
            a = np.array([1.0, 1.0], dtype=np.float32)
        else:
            a = np.array([0.0, 0.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(a)

        print()
        print("observation", observation)
        print("reward", reward)
        print("terminated", terminated)
        print("truncated", truncated)
        print("info", info)

        env.render()
    env.close()
