import os
import glob
import random
from typing import List, Optional
from pathlib import Path


import open3d
import traceback
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from rlbench.demo import Demo
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from utils.utils_without_rlbench import TASK_TO_ID


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class Mover:

    def __init__(self, task, disabled=False, max_tries=1):
        self._task = task
        self._last_action = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled

    def __call__(self, action, collision_checking=False):
        if self._disabled:
            return self._task.step(action)

        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()

        images = []
        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            obs, reward, terminate, other_obs = self._task.step(
                action, collision_checking=collision_checking)
            if other_obs == []:
                other_obs = [obs]
            for o in other_obs:
                images.append(
                    {
                        k.split("_")[0]: getattr(o, k)
                        for k in o.__dict__.keys()
                        if "_rgb" in k and getattr(o, k) is not None
                    }
                )

            pos = obs.gripper_pose[:3]
            rot = obs.gripper_pose[3:7]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            dist_rot = np.sqrt(np.square(target[3:7] - rot).sum())
            criteria = (dist_pos < 5e-3,)

            if all(criteria) or reward == 1:
                break

            print(
                f"Too far away (pos: {dist_pos:.3f}, rot: {dist_rot:.3f}, step: {self._step_id})... Retrying..."
            )

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            obs, reward, terminate, other_obs = self._task.step(action)
            if other_obs == []:
                other_obs = [obs]
            for o in other_obs:
                images.append(
                    {
                        k.split("_")[0]: getattr(o, k)
                        for k in o.__dict__.keys()
                        if "_rgb" in k and getattr(o, k) is not None
                    }
                )

        if try_id == self._max_tries:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()

        return obs, reward, terminate, images


class Actioner:

    def __init__(
        self,
        keypose_model=None,
        traj_model=None,
        instructions=None,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        action_dim=7,
        predict_keypose=True,
        predict_trajectory=False
    ):
        self._keypose_model = keypose_model
        self._traj_model = traj_model
        self._instructions = instructions
        self._apply_cameras = apply_cameras
        self._action_dim = action_dim
        self._predict_keypose = predict_keypose
        self._predict_trajectory = predict_trajectory

        self._actions = {}
        self._instr = None
        self._task_str = None

        if predict_keypose:
            assert keypose_model is not None
            self._keypose_model.eval()
        if predict_trajectory:
            assert traj_model is not None
            self._traj_model.eval()

    def load_episode(self, task_str, variation):
        self._task_str = task_str
        instructions = list(self._instructions[task_str][variation])
        self._instr = random.choice(instructions).unsqueeze(0)
        self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}

    def get_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)

        action_ls = []
        trajectory_ls = []
        for i in range(len(key_frame)):
            obs = demo[key_frame[i]]
            action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
            action = torch.from_numpy(action_np)
            action_ls.append(action.unsqueeze(0))

            trajectory_np = []
            for j in range(key_frame[i - 1] if i > 0 else 0, key_frame[i]):
                obs = demo[j]
                trajectory_np.append(np.concatenate([
                    obs.gripper_pose, [obs.gripper_open]
                ]))
            trajectory_ls.append(np.stack(trajectory_np))

        trajectory_mask_ls = [
            torch.zeros(1, key_frame[i] - (key_frame[i - 1] if i > 0 else 0)).bool()
            for i in range(len(key_frame))
        ]

        return action_ls, trajectory_ls, trajectory_mask_ls

    def predict(self, rgbs, pcds, gripper, gt_action, trajectory_mask):
        output = {"action": None, "attention": {}}

        # Fix order of views for HiveFormer
        rgbs = rgbs / 2 + 0.5  # in [0, 1]

        if self._instr is None:
            raise ValueError()

        self._instr = self._instr.to(rgbs.device)
        self._task_id = self._task_id.to(rgbs.device)

        # Predict keypose
        if self._predict_keypose:
            print('Predict Keypose')
            pred = self._keypose_model(
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[:, -1, :self._action_dim],
            )
            output["action"] = torch.cat(
                [pred["position"], pred["rotation"], pred["gripper"]],
                dim=1
            )
        else:
            output["action"] = gt_action[:, -1]

        # Predict trajectory
        if self._predict_trajectory:
            print('Predict Trajectory')
            output["trajectory"] = self._traj_model.compute_trajectory(
                trajectory_mask,
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[:, -1, :self._action_dim],
                output["action"][..., :self._action_dim]
            )
        else:
            output["trajectory"] = None

        return output

    @property
    def device(self):
        if self._keypose_model is not None:
            return next(self._keypose_model.parameters()).device
        if self._traj_model is not None:
            return next(self._traj_model.parameters()).device


def obs_to_attn(obs, camera):
    extrinsics_44 = torch.from_numpy(
        obs.misc[f"{camera}_camera_extrinsics"]
    ).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(
        obs.misc[f"{camera}_camera_intrinsics"]
    ).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        traj_cmd=False,
        exec168=False,
        image_size=(128, 128),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        fine_sampling_ball_diameter=None,
        collision_checking=False
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        if traj_cmd:
            self.action_mode = MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
                gripper_action_mode=Discrete()
            )
        else:
            self.action_mode = MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
                gripper_action_mode=Discrete()
            )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )
        self.image_size = image_size
        self.exec168 = exec168

    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        state = transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        attns = torch.Tensor([])
        for cam in self.apply_cameras:
            u, v = obs_to_attn(obs, cam)
            attn = torch.zeros(1, 1, 1, self.image_size[0], self.image_size[1])
            if not (u < 0 or u > self.image_size[1] - 1 or v < 0 or v > self.image_size[0] - 1):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)

        return rgb, pcd, gripper

    def get_obs_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)
        key_frame.insert(0, 0)
        state_ls = []
        action_ls = []
        for f in key_frame:
            state, action = self.get_obs_action(demo._observations[f])
            state = transform(state, augmentation=False)
            state_ls.append(state.unsqueeze(0))
            action_ls.append(action.unsqueeze(0))
        return state_ls, action_ls

    def get_gripper_matrix_from_action(self, action):
        action = action.cpu().numpy()
        position = action[:3]
        quaternion = action[3:7]
        rotation = open3d.geometry.get_rotation_matrix_from_quaternion(
            np.array((quaternion[3], quaternion[0], quaternion[1], quaternion[2]))
        )
        gripper_matrix = np.eye(4)
        gripper_matrix[:3, :3] = rotation
        gripper_matrix[:3, 3] = position
        return gripper_matrix

    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False
        )
        return demos

    def evaluate_task_on_multiple_variations(
        self,
        task_str: str,
        max_steps: int,
        num_variations: int,  # -1 means all variations
        num_demos: int,
        log_dir: Optional[Path],
        actioner: Actioner,
        max_tries: int = 1,
        save_attn: bool = False,
        record_videos: bool = False,
        num_videos: int = 10,
        record_demo_video: bool = False,
        offline: int = 0,
        position_prediction_only: bool = False,
        verbose: bool = False,
        dense_interpolation=False,
        interpolation_length=100,
        act3d_gripper_input_history=1,
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = task.variation_count()

        if num_variations > 0:
            task_variations = np.minimum(num_variations, task_variations)
            task_variations = range(task_variations)
        else:
            task_variations = glob.glob(os.path.join(self.data_path, task_str, "variation*"))
            task_variations = [int(n.split('/')[-1].replace('variation', '')) for n in task_variations]

        var_success_rates = {}
        var_num_valid_demos = {}

        #for variation in range(task_variations):
        for variation in task_variations:
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = self._evaluate_task_on_one_variation(
                task_str=task_str,
                task=task,
                max_steps=max_steps,
                variation=variation,
                num_demos=num_demos // len(task_variations) + 1,
                log_dir=log_dir,
                actioner=actioner,
                max_tries=max_tries,
                save_attn=save_attn,
                record_videos=record_videos,
                num_videos=num_videos,
                record_demo_video=record_demo_video,
                offline=offline,
                position_prediction_only=position_prediction_only,
                verbose=verbose,
                dense_interpolation=dense_interpolation,
                interpolation_length=interpolation_length,
                act3d_gripper_input_history=act3d_gripper_input_history
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = sum(var_success_rates.values()) / sum(var_num_valid_demos.values())

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str: str,
        task: TaskEnvironment,
        max_steps: int,
        variation: int,
        num_demos: int,
        log_dir: Optional[Path],
        actioner: Actioner,
        max_tries: int = 1,
        save_attn: bool = False,
        record_videos: bool = False,
        num_videos: int = 10,
        record_demo_video: bool = False,
        offline: int = 0,
        position_prediction_only: bool = False,
        verbose: bool = False,
        dense_interpolation=False,
        interpolation_length=50,
        act3d_gripper_input_history=1,
    ):
        device = actioner.device

        success_rate = 0
        num_valid_demos = 0
        total_reward = 0

        for demo_id in range(num_demos):
            if verbose:
                print()
                print(f"Starting demo {demo_id}")

            try:
                demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
                num_valid_demos += 1
            except:
                continue

            rgbs = torch.Tensor([]).to(device)
            pcds = torch.Tensor([]).to(device)
            grippers = torch.Tensor([]).to(device)

            # descriptions, obs = task.reset()
            descriptions, obs = task.reset_to_demo(demo)

            actioner.load_episode(task_str, variation)

            move = Mover(task, max_tries=max_tries)
            reward = None
            max_reward = 0.0
            gt_keyframe_actions, _, gt_trajectory_masks = actioner.get_action_from_demo(demo)
            if offline:
                max_steps = len(gt_keyframe_actions)

            for step_id in range(max_steps):

                # Fetch the current observation, and predict one action
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgb = rgb.to(device)
                pcd = pcd.to(device)
                gripper = gripper.to(device)

                rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
                pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)
                if dense_interpolation:
                    trajectory_mask = torch.full([1, interpolation_length], False).to(device)
                elif self.exec168:
                    trajectory_mask = torch.full([1, 16], False).to(device)
                else:
                    trajectory_mask = gt_trajectory_masks[step_id].to(device)

                # Prepare inputs to the keypose and trajectory model
                rgbs_input = rgbs[:, -1:][:, :, :, :3]
                pcds_input = pcds[:, -1:]
                gripper_input = grippers[:, -1:]
                if not actioner._predict_keypose:
                    gt_action_input = (
                        gt_keyframe_actions[step_id] if not self.exec168
                        else gt_keyframe_actions[-1]
                    ).unsqueeze(0).float().to(device)
                else:
                    gt_action_input = None

                # assert gt_action_input is None
                output = actioner.predict(
                    rgbs_input, pcds_input,
                    gripper_input,
                    gt_action=gt_action_input,
                    trajectory_mask=trajectory_mask
                )

                # Using ground-truth keypose `online` is True; otherwise, use predicted keypose
                if offline:
                    print('Setting action to ground-truth')
                    # Follow demo
                    action = gt_keyframe_actions[step_id] if not self.exec168 else gt_keyframe_actions[-1]
                    action[..., -1] = torch.round(action[..., -1])
                    output["action"] = action
                else:
                    # Follow trained policy
                    action = output["action"]
                    action[..., -1] = torch.round(action[..., -1])

                    if position_prediction_only:
                        print('Setting rotation to ground-truth')
                        action[:, 3:] = gt_keyframe_actions[step_id][:, 3:]
                    output["action"] = action

                if verbose:
                    print(f"Step {step_id}", action)
                if self.exec168:
                    output['trajectory'] = output['trajectory'][:, :8]

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    if output.get("trajectory", None) is not None:
                        trajectory_np = output["trajectory"][-1].cpu().numpy()

                        # append gripper action and next step
                        trajectory_np_full = trajectory_np
                        is_full = trajectory_np_full.shape[-1] == 8
                        # if gripper openess not predicted by trajectory model, move the gripper
                        # first and then open/close gripper at the last step.
                        if not is_full:
                            trajectory_np_full = np.concatenate([
                                trajectory_np,
                                np.tile(
                                    grippers[-1, -1:, -1:].cpu().numpy(),
                                    [trajectory_np.shape[0], 1]
                                )
                            ], axis=-1)
                            trajectory_np_full[-1, -1] = output['action'][-1, -1]
                        trajectory_np_full[:, -1] = trajectory_np_full[:, -1].round()

                        # execute
                        for action_np in tqdm(trajectory_np_full[1:]):
                            try:
                                obs, reward, terminate, _ = move(action_np)
                            except:
                                pass

                    # Or plan to reach next predicted keypoint
                    else:
                        print("Plan with RRT")
                        action_np = action[-1].detach().cpu().numpy()
                        collision_checking = self._collision_checking(task_str, step_id)
                        obs, reward, terminate, _ = move(action_np, collision_checking=collision_checking)

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                    if terminate:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0
                    break

            total_reward += max_reward
            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}",
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", num_valid_demos,
            )

        # Compensate for failed demos
        if num_valid_demos == 0:
            assert success_rate == 0
            valid = False
        else:
            valid = True

        return success_rate, valid, num_valid_demos

    def _collision_checking(self, task_str, step_id):
        """Collision checking for planner."""
        # collision_checking = True
        collision_checking = False
        # if task_str == 'close_door':
        #     collision_checking = True
        # if task_str == 'open_fridge' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'open_oven' and step_id == 3:
        #     collision_checking = True
        # if task_str == 'hang_frame_on_hanger' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'take_frame_off_hanger' and step_id == 0:
        #     for i in range(300):
        #         self.env._scene.step()
        #     collision_checking = True
        # if task_str == 'put_books_on_bookshelf' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'slide_cabinet_open_and_place_cups' and step_id == 0:
        #     collision_checking = True
        return collision_checking

    def verify_demos(
        self,
        task_str: str,
        variation: int,
        num_demos: int,
        max_tries: int = 1,
        verbose: bool = False,
    ):
        if verbose:
            print()
            print(f"{task_str}, variation {variation}, {num_demos} demos")

        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task.set_variation(variation)  # type: ignore

        success_rate = 0.0
        invalid_demos = 0

        for demo_id in range(num_demos):
            if verbose:
                print(f"Starting demo {demo_id}")

            try:
                demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
            except:
                print(f"Invalid demo {demo_id} for {task_str} variation {variation}")
                print()
                traceback.print_exc()
                invalid_demos += 1

            task.reset_to_demo(demo)

            gt_keyframe_actions = []
            for f in keypoint_discovery(demo):
                obs = demo[f]
                action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
                gt_keyframe_actions.append(action)

            move = Mover(task, max_tries=max_tries)

            for step_id, action in enumerate(gt_keyframe_actions):
                if verbose:
                    print(f"Step {step_id}")

                try:
                    obs, reward, terminate, step_images = move(action)
                    if reward == 1:
                        success_rate += 1 / num_demos
                        break
                    if terminate and verbose:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_type, demo, success_rate, e)
                    reward = 0
                    break

            if verbose:
                print(f"Finished demo {demo_id}, SR: {success_rate}")

        # Compensate for failed demos
        if (num_demos - invalid_demos) == 0:
            success_rate = 0.0
            valid = False
        else:
            success_rate = success_rate * num_demos / (num_demos - invalid_demos)
            valid = True

        self.env.shutdown()
        return success_rate, valid, invalid_demos

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        unused_cams = CameraConfig()
        unused_cams.set_all(False)
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            **kwargs,
        )

        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )

        return obs_config


# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer, delta):
    next_is_not_final = i == (len(demo) - 2)
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints


def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        if augmentation:
            raise NotImplementedError()  # Deprecated

        # normalise to [-1, 1]
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    obs = obs_rgb + obs_depth + obs_pc
    return torch.cat(obs, dim=0)
