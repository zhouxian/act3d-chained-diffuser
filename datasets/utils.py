import blosc
import pickle

import einops
from pickle import UnpicklingError
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import utils.pytorch3d_transforms as torch3d_tf

from model.utils.utils import normalise_quat


def loader(file):
    if str(file).endswith(".npy"):
        try:
            content = np.load(file, allow_pickle=True)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".dat"):
        try:
            with open(file, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".pkl"):
        try:
            with open(file, 'rb') as f:
                content = pickle.load(f)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    return None


class Resize:
    """
    Resize and pad/crop the image and aligned point cloud.
    """
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs):
        """
        Accept tensors as T, N, C, H, W
        """
        keys = list(kwargs.keys())

        if len(keys) == 0:
            raise RuntimeError("No args")

        # Sample resize scale from continuous range
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize
        kwargs = {
            n: transforms_f.resize(
                arg,
                resized_size,
                transforms.InterpolationMode.NEAREST
            )
            for n, arg in kwargs.items()
        }

        # If resized image is smaller than original, pad it with a reflection
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad, bottom_pad = max(raw_w - resized_size[1], 0), max(
                raw_h - resized_size[0], 0
            )
            kwargs = {
                n: transforms_f.pad(
                    arg,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="reflect",
                )
                for n, arg in kwargs.items()
            }

        # If resized image is larger than original, crop it
        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )
        kwargs = {
            n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()
        }

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs


class Rotate:
    """
    Rotate point cloud, current gripper, and next ground-truth gripper, while
    ensuring current/next ground-truth gripper stay within workspace bounds.
    """
    def __init__(self, gripper_loc_bounds, yaw_range, num_tries=10):
        self.gripper_loc_bounds = torch.from_numpy(gripper_loc_bounds)
        self.yaw_range = np.deg2rad(yaw_range)
        self.num_tries = num_tries

    def __call__(self, pcds, gripper, action, mask, trajectory=None):
        if self.yaw_range == 0.0:
            return pcds, gripper, action, trajectory

        augmentation_rot_4x4 = self._sample_rotation()
        gripper_rot_4x4 = self._gripper_action_to_matrix(gripper)
        action_rot_4x4 = self._gripper_action_to_matrix(action)
        if trajectory is not None:
            trajectory = einops.rearrange(trajectory, 'b l c -> (b l ) c')
            traj_rot = self._gripper_action_to_matrix(trajectory)

        for _ in range(self.num_tries):
            gripper_rot_4x4 = augmentation_rot_4x4 @ gripper_rot_4x4
            action_rot_4x4 = augmentation_rot_4x4 @ action_rot_4x4
            # p: position, q: quaternion
            gripper_p, gripper_q = self._gripper_matrix_to_action(gripper_rot_4x4)
            action_p, action_q = self._gripper_matrix_to_action(action_rot_4x4)
            if trajectory is not None:
                traj_p, traj_q = self._gripper_matrix_to_action(traj_rot)

            if self._check_bounds(gripper_p[mask], action_p[mask]):
                gripper[mask, :3], gripper[mask, 3:7] = gripper_p[mask], gripper_q[mask]
                action[mask, :3], action[mask, 3:7] = action_p[mask], action_q[mask]
                if trajectory is not None:
                    trajectory[:, :3], trajectory[:, 3:7] = traj_p, traj_q
                    trajectory = trajectory.reshape(
                        len(action), -1, trajectory.size(-1)
                    )
                pcds[mask] = einops.einsum(
                    augmentation_rot_4x4[:3, :3], pcds[mask],
                    "c2 c1, t ncam c1 h w -> t ncam c2 h w"
                )
                break

        return pcds, gripper, action, trajectory

    def _check_bounds(self, gripper_position, action_position):
        return (
            (gripper_position >= self.gripper_loc_bounds[0]).all() and
            (gripper_position <= self.gripper_loc_bounds[1]).all() and
            (action_position >= self.gripper_loc_bounds[0]).all() and
            (action_position <= self.gripper_loc_bounds[1]).all()
        )

    def _sample_rotation(self):
        yaw = 2 * self.yaw_range * torch.rand(1) - self.yaw_range
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        rot_3x3 = torch3d_tf.euler_angles_to_matrix(
            torch.stack([roll, pitch, yaw], dim=1), "XYZ"
        )
        rot_4x4 = torch.eye(4)
        rot_4x4[:3, :3] = rot_3x3
        return rot_4x4

    def _gripper_action_to_matrix(self, action):
        position = action[:, :3]
        quaternion = action[:, [6, 3, 4, 5]]
        rot_3x3 = torch3d_tf.quaternion_to_matrix(quaternion)
        rot_4x4 = torch.eye(4).unsqueeze(0).repeat(position.shape[0], 1, 1)
        rot_4x4[:, :3, :3] = rot_3x3
        rot_4x4[:, :3, 3] = position
        return rot_4x4

    def _gripper_matrix_to_action(self, matrix):
        position = matrix[:, :3, 3]
        rot_3x3 = matrix[:, :3, :3]
        quaternion = torch3d_tf.matrix_to_quaternion(rot_3x3)[:, [1, 2, 3, 0]]
        return position, quaternion


class TrajectoryInterpolator:
    """Interpolate a trajectory to have fixed length."""

    def __init__(self, use=False, interpolation_length=50):
        self._use = use
        self._interpolation_length = interpolation_length

    def __call__(self, trajectory):
        if not self._use:
            return trajectory
        trajectory = trajectory.numpy()
        # Calculate the current number of steps
        old_num_steps = len(trajectory)

        # Create a 1D array for the old and new steps
        old_steps = np.linspace(0, 1, old_num_steps)
        new_steps = np.linspace(0, 1, self._interpolation_length)

        # Interpolate each dimension separately
        resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            if i == 7:  # gripper opening
                interpolator = interp1d(old_steps, trajectory[:, i])
            else:
                interpolator = CubicSpline(old_steps, trajectory[:, i])

            resampled[:, i] = interpolator(new_steps)

        resampled = torch.tensor(resampled)
        resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
        return resampled
