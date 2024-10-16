from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections.abc import Sequence
from typing import Optional
import copy

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
)
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.actuators import ImplicitActuatorCfg


def change_joint_properties(
    articulation: Articulation,
    act: str,
    stiffness: Optional[float] = None,
    damping: Optional[float] = None,
    friction: Optional[float] = None,
    armature: Optional[float] = None,
    verbose: bool = False,
) -> None:
    """Change the joint properties of the actuator group in the articulation.
    args:
    - articulation: Articulation object
    - act: actuator group name
    - stiffness: new stiffness value
    - damping: new damping value
    - friction: new friction value
    - armature: new armature value
    - verbose: print joint properties before and after change

    returns: None
    """

    current_stiffness = articulation.actuators[act].stiffness
    current_damping = articulation.actuators[act].damping
    current_friction = articulation.actuators[act].friction
    current_armature = articulation.actuators[act].armature
    if verbose:
        print(f"[INFO]: changing joint properties for actuator group {act}...")
        print("joint properties before change:")
        print("stiffness: ", current_stiffness)
        print("damping: ", current_damping)
        print("friction: ", current_friction)
        print("armature: ", current_armature)

    if stiffness is not None:
        new_stiffness_tensor = (
            torch.ones_like(current_stiffness, device=current_stiffness.device)
            * stiffness
        )
        articulation.actuators[act].stiffness = new_stiffness_tensor
        articulation.write_joint_stiffness_to_sim(
            stiffness=new_stiffness_tensor,
            joint_ids=articulation.actuators[act].joint_indices,
        )
    if damping is not None:
        new_damping_tensor = (
            torch.ones_like(current_damping, device=current_damping.device) * damping
        )
        articulation.actuators[act].damping = new_damping_tensor
        articulation.write_joint_damping_to_sim(
            damping=new_damping_tensor,
            joint_ids=articulation.actuators[act].joint_indices,
        )
    if friction is not None:
        new_friction_tensor = (
            torch.ones_like(current_friction, device=current_friction.device) * friction
        )
        articulation.actuators[act].friction = new_friction_tensor
        articulation.write_joint_friction_to_sim(
            joint_friction=new_friction_tensor,
            joint_ids=articulation.actuators[act].joint_indices,
        )
    if armature is not None:
        new_armature_tensor = (
            torch.ones_like(current_armature, device=current_armature.device) * armature
        )
        articulation.actuators[act].armature = new_armature_tensor
        articulation.write_joint_armature_to_sim(
            armature=new_armature_tensor,
            joint_ids=articulation.actuators[act].joint_indices,
        )

    current_stiffness = articulation.actuators[act].stiffness
    current_damping = articulation.actuators[act].damping
    current_friction = articulation.actuators[act].friction
    current_armature = articulation.actuators[act].armature
    if verbose:
        print("joint properties after change:")
        print("stiffness: ", current_stiffness)
        print("damping: ", current_damping)
        print("friction: ", current_friction)
        print("armature: ", current_armature)


@configclass
class FrankaReachCustomEnvCfg(DirectRLEnvCfg):
    # environment variables
    episode_length_s = 3.0
    target_reach_threshold = 0.1  # probably less than this
    active_joints_idx = [0, 1, 2, 3, 4, 5]
    num_actions: int = len(active_joints_idx)
    num_observations = 20
    num_states = 0
    decimation = 1
    control = "position"
    joint_stiffness = 400.0
    joint_damping = 80.0
    joint_friction = 1.07
    joint_armature = 0.01

    # simulation
    sim: SimulationCfg = SimulationCfg(
        disable_contact_processing=True,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=3.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.0,
                "panda_finger_joint.*": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=joint_stiffness,
                damping=joint_damping,
                friction=joint_friction,
                armature=joint_armature,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=joint_stiffness,
                damping=joint_damping,
                friction=joint_friction,
                armature=joint_armature,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # target
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0, disable_gravity=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )

    def set_active_joints_idx(self, active_joints_idx) -> None:
        self.active_joints_idx = active_joints_idx
        self.num_actions = len(self.active_joints_idx)


class FrankaReachCustomEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaReachCustomEnvCfg

    def __init__(
        self, cfg: FrankaReachCustomEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.terminate_on_target = False

        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (self.num_envs, self.cfg.num_actions)
        )

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(
            device=self.device
        )
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(
            device=self.device
        )
        self.robot_dof_velocity_limits = self.robot.data.soft_joint_vel_limits[0].to(
            device=self.device
        )
        self.robot_dof_effort_limits = torch.zeros_like(self.robot_dof_velocity_limits)

        self.robot_dof_pos_targets = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        self.robot_dof_vel_targets = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        self.robot_dof_acc_targets = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        self.robot_dof_eff_targets = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        self.robot_joint_pos = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        self.robot_joint_vel = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        self.robot_joint_acc = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        self.robot_joint_tor_comp = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        self.robot_joint_tor_applied = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        self.robot_ee_pos = torch.zeros(
            (self.num_envs, 3), device=self.device
        )  # end effector position
        self.target_pos = torch.zeros(
            (self.num_envs, 3), device=self.device
        )  # target position
        self.distance_ee_target = torch.zeros(self.num_envs, device=self.device)
        self.runtime = torch.zeros(self.num_envs, device=self.device)
        self.extras = {
            "runtime": self.runtime,
        }
        # make sure the active_joints_idx
        # 0. are ordered in ascending order
        # 1. no more than 7 joints
        # 2. no less than 1 joint
        # 3. no duplicate joints
        # 4. no joints outside the range [0 - 6]
        self.cfg.active_joints_idx.sort()
        assert len(self.cfg.active_joints_idx) <= 7
        assert len(self.cfg.active_joints_idx) >= 1
        assert len(self.cfg.active_joints_idx) == len(set(self.cfg.active_joints_idx))
        assert all(
            [
                joint_idx >= 0 and joint_idx <= 6
                for joint_idx in self.cfg.active_joints_idx
            ]
        )
        self.active_joints_idx = torch.tensor(
            self.cfg.active_joints_idx, device=self.device
        )

        self.post_init()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.target = RigidObject(self.cfg.target)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.target
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def post_init(self):

        # change joint properties
        # TODO: have these values come from the config file
        if self.cfg.control == "position":
            stiffness = 400.0
            damping = 80.0
        elif self.cfg.control == "velocity":
            stiffness = 400.0
            damping = 80.0
        elif self.cfg.control == "acceleration":
            stiffness = 400.0
            damping = 80.0
        elif self.cfg.control == "effort":
            stiffness = 0.0
            damping = 0.0
        acts = ["panda_shoulder", "panda_forearm"]
        for act in acts:
            change_joint_properties(
                self.robot,
                act=act,
                stiffness=stiffness,
                damping=damping,
                friction=1.0,
                armature=0.01,
            )

        # get the robot effort limits
        for _, act in self.robot.actuators.items():
            joint_idx = act.joint_indices
            self.robot_dof_effort_limits[joint_idx] = act.effort_limit[0]
        print(f"[INFO]: robot dof effort limits: {self.robot_dof_effort_limits}")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        self.actions = actions.clone().to(self.device)
        self.actions.clamp_(-1.0, 1.0)

        if self.cfg.control == "position":
            # scale the actions within the joint position limits [self.robot_dof_lower_limits, self.robot_dof_upper_limits]
            self.robot_dof_pos_targets[:, self.active_joints_idx] = (
                0.5
                * (self.actions + 1.0)
                * (
                    self.robot_dof_upper_limits[self.active_joints_idx]
                    - self.robot_dof_lower_limits[self.active_joints_idx]
                )
                + self.robot_dof_lower_limits[self.active_joints_idx]
            )
        elif self.cfg.control == "velocity":
            # scale the actions within the joint velocity limits [-self.robot_dof_velocity_limits, self.robot_dof_velocity_limits]
            self.robot_dof_vel_targets[:, self.active_joints_idx] = (
                self.actions * self.robot_dof_velocity_limits[self.active_joints_idx]
            )
            self.robot_dof_pos_targets[:, self.active_joints_idx] = (
                self.robot_dof_pos_targets[:, self.active_joints_idx]
                # self.robot_joint_pos[:, self.active_joints_idx]
                + self.robot_dof_vel_targets[:, self.active_joints_idx] * self.step_dt
            ).clamp(
                self.robot_dof_lower_limits[self.active_joints_idx],
                self.robot_dof_upper_limits[self.active_joints_idx],
            )
        elif self.cfg.control == "acceleration":
            # ! action scaling is not known for acceleration control
            self.robot_dof_acc_targets[:, self.active_joints_idx] = self.actions
            self.robot_dof_vel_targets[:, self.active_joints_idx] = (
                self.robot_dof_vel_targets[:, self.active_joints_idx]
                # self.robot_joint_vel[:, self.active_joints_idx]
                + self.robot_dof_acc_targets[:, self.active_joints_idx] * self.step_dt
            ).clamp(
                -self.robot_dof_velocity_limits[self.active_joints_idx],
                self.robot_dof_velocity_limits[self.active_joints_idx],
            )
            self.robot_dof_pos_targets[:, self.active_joints_idx] = (
                self.robot_dof_pos_targets[:, self.active_joints_idx]
                # self.robot_joint_pos[:, self.active_joints_idx]
                + self.robot_dof_vel_targets[:, self.active_joints_idx] * self.step_dt
            ).clamp(
                self.robot_dof_lower_limits[self.active_joints_idx],
                self.robot_dof_upper_limits[self.active_joints_idx],
            )
        elif self.cfg.control == "effort":
            # scale the actions within the joint effort limits [-self.robot_dof_effort_limits, self.robot_dof_effort_limits]
            self.robot_dof_eff_targets[:, self.active_joints_idx] = (
                self.actions * self.robot_dof_effort_limits[self.active_joints_idx]
            )

        # record the runtime update
        self.runtime += self.step_dt
        self.extras = {
            "runtime": self.runtime,
        }

    def _apply_action(self) -> None:
        """print(f"[INFO]: applying action in control mode '{self.cfg.control}'...")
        print(f"[INFO]: robot dof pos targets: {self.robot_dof_pos_targets[0]}")
        print(f"[INFO]: robot dof vel targets: {self.robot_dof_vel_targets[0]}")
        print(f"[INFO]: robot dof acc targets: {self.robot_dof_acc_targets[0]}")
        print(f"[INFO]: robot dof eff targets: {self.robot_dof_eff_targets[0]}")
        print(f"[INFO]: robot dof pos: {self.robot_joint_pos[0]}")
        print(f"[INFO]: robot dof vel: {self.robot_joint_vel[0]}")
        print(f"[INFO]: robot dof acc: {self.robot_joint_acc[0]}")
        print(f"[INFO]: robot dof torque comp: {self.robot_joint_tor_comp[0]}")
        print(f"[INFO]: robot dof torque applied: {self.robot_joint_tor_applied[0]}")"""
        if self.cfg.control == "effort":
            self.robot.set_joint_effort_target(self.robot_dof_eff_targets)
        else:
            self.robot.set_joint_position_target(self.robot_dof_pos_targets)
            self.robot.set_joint_velocity_target(self.robot_dof_vel_targets)

    def _get_observations(self) -> dict:
        # transform robot joint positions and velocity to [-1, 1]
        # we do not care about the finger tips for this task so we remove the last two joints
        robot_joint_pos_scaled = (
            2.0
            * (self.robot_joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )[:, :-2]
        robot_joint_vel_scaled = (
            self.robot_joint_vel / self.robot_dof_velocity_limits
        )[:, :-2]

        obs = torch.cat(
            (
                robot_joint_pos_scaled,
                robot_joint_vel_scaled,
                self.robot_ee_pos,
                self.target_pos,
            ),
            dim=-1,
        )
        return {
            "policy": obs,
        }

    def _get_rewards(self) -> torch.Tensor:
        return -self.distance_ee_target

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        terminated = self.check_termination()
        truncated = self.runtime + 0.0001 >= self.max_episode_length_s
        dones = terminated | truncated
        if dones.any():
            self.extras["_final_observation"] = dones
            self.extras["final_observation"] = [
                {"policy": obs} for obs in self._get_observations()["policy"]
            ]
        return terminated, truncated

    def check_termination(self) -> torch.Tensor:

        # check if the end effector is within a certain distance of the target
        target_reached = self.distance_ee_target < self.cfg.target_reach_threshold

        if self.terminate_on_target:
            return target_reached
        return torch.zeros_like(target_reached, dtype=torch.bool)

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.last_observation = self._get_observations()

        # get random valid robot joint positions
        self.reset_joints_by_sampling_in_limits(env_ids)
        # set target pos to end effector pos
        self.set_target_pos_to_ee_pos(env_ids)
        # get new actual robot joint positions
        self.reset_joints_by_sampling_in_limits(env_ids)
        # reset the runtime
        self.runtime[env_ids] = 0.0

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def reset_joints_by_sampling_in_limits(
        self,
        env_ids: torch.Tensor,
    ):
        """Reset the robot joints by sampling within the position limits and setting the velocity to zero."""

        # get joint pos limits
        joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos_shape = default_joint_pos.shape

        # sample between these values randomly
        sample_joint_pos = sample_uniform(
            joint_pos_limits[..., 0],
            joint_pos_limits[..., 1],
            joint_pos_shape,
            joint_pos_limits.device,
        )

        # set the joint pos to the sampled values for the active joints
        new_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        new_joint_pos[:, self.active_joints_idx] = sample_joint_pos[
            :, self.active_joints_idx
        ]

        # set joint vel to zero
        new_joint_vel = torch.zeros_like(new_joint_pos, device=new_joint_pos.device)

        # set into the physics simulation
        self.robot.write_joint_state_to_sim(
            new_joint_pos, new_joint_vel, env_ids=env_ids
        )
        self.robot.set_joint_position_target(new_joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocity_target(new_joint_vel, env_ids=env_ids)
        self.robot.write_data_to_sim()

        # ! IMPORTANT: running the sim for a very small time step to update robot.data.body_pos_w
        # quite hacky but it works... for now
        self.robot.update(1e-10)
        self.sim._physics_context._physx_sim_interface.simulate(
            1e-10, self.sim._current_time
        )
        self.sim._physics_context._physx_sim_interface.fetch_results()

        # update local variables
        self.robot_joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
        self.robot_joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        self.robot_dof_pos_targets[env_ids] = self.robot_joint_pos[env_ids]
        self.robot_dof_vel_targets[env_ids] = self.robot_joint_vel[env_ids]
        # update end effector position
        env_pos = self.robot.data.body_pos_w[env_ids, 0]
        left_finger_pos = self.robot.data.body_pos_w[env_ids, -2]
        right_finger_pos = self.robot.data.body_pos_w[env_ids, -1]
        self.robot_ee_pos[env_ids] = (
            0.5 * (left_finger_pos + right_finger_pos) - env_pos
        )

    def set_target_pos_to_ee_pos(self, env_ids: torch.Tensor):
        """Set the target position to the end effector position."""

        # set target pos to end effector pos
        env_pos = self.robot.data.body_pos_w[env_ids, 0]
        new_target_pos = self.robot_ee_pos[env_ids] + env_pos
        new_target_pose = torch.cat(
            (
                new_target_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                    env_ids.shape[0], 1
                ),
            ),
            dim=-1,
        )
        self.target.write_root_pose_to_sim(new_target_pose, env_ids)
        self.target.reset(env_ids)
        self.target_pos[env_ids] = self.target.data.body_pos_w[env_ids, 0] - env_pos

    def _compute_intermediate_values(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # robot state
        self.robot_joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
        self.robot_joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        self.robot_joint_acc[env_ids] = self.robot.data.joint_acc[env_ids]
        self.robot_joint_tor_comp[env_ids] = self.robot.data.computed_torque[env_ids]
        self.robot_joint_tor_applied[env_ids] = self.robot.data.applied_torque[env_ids]

        # get each env root pos
        env_pos = self.robot.data.body_pos_w[env_ids, 0]

        # end effector position
        left_finger_pos = self.robot.data.body_pos_w[env_ids, -2]
        right_finger_pos = self.robot.data.body_pos_w[env_ids, -1]
        self.robot_ee_pos[env_ids] = (
            0.5 * (left_finger_pos + right_finger_pos) - env_pos
        )

        # target state
        assert self.target.data.body_pos_w.shape[1] == 1
        self.target_pos[env_ids] = self.target.data.body_pos_w[env_ids, 0] - env_pos

        self.distance_ee_target[env_ids] = torch.norm(
            self.robot_ee_pos[env_ids] - self.target_pos[env_ids], dim=-1
        )


class FrankaEnv(gym.vector.SyncVectorEnv):
    def __init__(
        self,
        seed: int = None,
        max_episode_steps: int = 200,
        render_mode: str = "human",
        dt: float = 0.02,
        eval: bool = False,
        num_envs: int = 1,
        render_interval: int = 1,
        active_joints: list[int] = [0, 1, 2, 3, 4, 5],
        control: str = "position",
        **kwargs,
    ) -> None:
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(1.0 / (dt * render_interval)),
        }

        self.dt = dt  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0
        self.render_mode = render_mode
        self.fully_observable = True  # pixel observations not supported at the moment
        self.num_envs = num_envs

        env_cfg = FrankaReachCustomEnvCfg()
        env_cfg.scene.num_envs = num_envs
        env_cfg.sim.dt = dt
        env_cfg.episode_length_s = max_episode_steps * dt * render_interval
        env_cfg.decimation = render_interval
        env_cfg.sim.render_interval = render_interval
        env_cfg.set_active_joints_idx(active_joints)
        env_cfg.control = control

        self._env = FrankaReachCustomEnv(
            cfg=env_cfg, render_mode="rgb_array", terminate_on_target=False
        )
        self.action_space = self._env.action_space

        obs_limits = np.tile(np.array([1.0] * 14 + [np.inf] * 3), (num_envs, 1))
        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(
                    low=-1.0 * obs_limits, high=obs_limits, shape=(num_envs, 17)
                ),
                "target": spaces.Box(
                    low=np.tile(np.array([-np.inf] * 3), (num_envs, 1)),
                    high=np.tile(np.array([np.inf] * 3), (num_envs, 1)),
                    shape=(num_envs, 3),
                ),
            }
        )

        self.state_labels = [
            "joint 1 pos",
            "joint 2 pos",
            "joint 3 pos",
            "joint 4 pos",
            "joint 5 pos",
            "joint 6 pos",
            "joint 7 pos",
            "joint 1 vel",
            "joint 2 vel",
            "joint 3 vel",
            "joint 4 vel",
            "joint 5 vel",
            "joint 6 vel",
            "joint 7 vel",
            "hand x",
            "hand y",
            "hand z",
        ]

        self.target_labels = [
            "hand x",
            "hand y",
            "hand z",
        ]

        self.loss_gain = {
            "gain": np.array([1.0, 1.0, 1.0]),
            "use": np.array([self.state_labels.index(l) for l in self.target_labels]),
        }

        self.manual_video = False

    def reset(self, *args, **kwargs):
        obs, extras = self._env.reset(*args, **kwargs)
        return self.redo_obs(obs), self.redo_extras(extras)

    def redo_extras(self, extras: dict):
        extras = copy.deepcopy(extras)
        if "_final_observation" in extras.keys():
            extras["final_observation"] = [
                {
                    "proprio": obs["policy"][:-3],
                    "target": obs["policy"][-3:],
                }
                for obs in extras["final_observation"]
            ]
        return extras

    def redo_obs(self, obs: dict):
        observations = {
            "proprio": obs["policy"][:, :-3],
            "target": obs["policy"][:, -3:],
        }
        return observations

    def step(self, action):
        # if action is a list or numpy array, convert it to a tensor
        if isinstance(action, (list, np.ndarray)):
            action = torch.tensor(action, dtype=torch.float32)
        obs, reward, terminated, truncated, extras = self._env.step(action)

        return self.redo_obs(obs), reward, terminated, truncated, self.redo_extras(extras)

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        self._env.seed(seed)

    def call(self, method: str, *args, **kwargs):

        # ! THIS IS A HOTFIX TO SAVE TIME AND IT IS BAD CODE DESIGN!
        if method.lower() == "get_loss_gain":
            return [self.loss_gain]

        return getattr(self._env, method)(*args, **kwargs)

    def get_attr(self, name: str):
        if name.lower() == "state_labels":
            return [self.state_labels]
        return getattr(self._env, name)

    @property
    def max_episode_steps(self):
        return self._env.max_episode_length

    @property
    def is_vector_env(self):
        return True