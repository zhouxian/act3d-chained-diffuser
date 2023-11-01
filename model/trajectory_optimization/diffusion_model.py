from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
from torch import nn
from torch.nn import functional as F

from model.utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat
)
import utils.pytorch3d_transforms as pytorch3d_transforms
from .diffusion_head import DiffusionHead


class DiffusionPlanner(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_vis_ins_attn_layers=2,
                 num_query_cross_attn_layers=8,
                 use_instruction=False,
                 use_goal=False,
                 use_goal_at_test=True,
                 feat_scales_to_use=1,
                 attn_rounds=1,
                 weight_tying=False,
                 gripper_loc_bounds=None,
                 rotation_parametrization='quat',
                 diffusion_timesteps=100):
        super().__init__()
        self._use_goal = use_goal
        self._use_goal_at_test = use_goal_at_test
        self._rotation_parametrization = rotation_parametrization
        self.prediction_head = DiffusionHead(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            use_instruction=use_instruction,
            use_goal=use_goal,
            feat_scales_to_use=feat_scales_to_use,
            attn_rounds=attn_rounds,
            weight_tying=weight_tying,
            rotation_parametrization=rotation_parametrization
        )
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample"
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample"
        )
        self.n_steps = diffusion_timesteps
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            goal_gripper
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            trajectory_mask,
            timestep,
            visible_rgb=rgb_obs,
            visible_pcd=pcd_obs,
            curr_gripper=curr_gripper,
            goal_gripper=goal_gripper,
            instruction=instruction
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        self.position_noise_scheduler.set_timesteps(self.n_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_steps)

        # Random trajectory, conditioned on start-end
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        trajectory = trajectory + condition_data

        # Iterative denoising
        timesteps = self.position_noise_scheduler.timesteps
        for t in timesteps:
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs
            )
            out = out[-1]  # keep only last layer's output
            out[condition_mask] = condition_data[condition_mask]
            if t == timesteps[-1]:
                trajectory = out
            else:
                pos = self.position_noise_scheduler.step(
                    out[..., :3], t, trajectory[..., :3]
                ).prev_sample
                rot = self.rotation_noise_scheduler.step(
                    out[..., 3:9], t, trajectory[..., 3:9]
                ).prev_sample
                trajectory = torch.cat((pos, rot), -1)

        return trajectory

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        goal_gripper
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        goal_gripper = goal_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[:, :3] = self.normalize_pos(curr_gripper[:, :3])
        goal_gripper[:, :3] = self.normalize_pos(goal_gripper[:, :3])
        curr_gripper = self.convert_rot(curr_gripper)
        goal_gripper = self.convert_rot(goal_gripper)

        # Prepare inputs
        fixed_inputs = (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            goal_gripper
        )

        # Condition on start-end pose
        B, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=rgb_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        # start pose
        cond_data[:, 0] = curr_gripper
        cond_mask[:, 0] = 1
        # end pose
        if self._use_goal_at_test:
            for d in range(len(cond_data)):
                neg_len_ = -trajectory_mask[d].sum().long()
                cond_data[d][neg_len_ - 1] = goal_gripper[d]
                cond_mask[d][neg_len_ - 1:] = 1
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs
        )

        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])

        return trajectory

    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def convert_rot(self, signal):
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        if self._rotation_parametrization == '6D':
            rot = pytorch3d_transforms.quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            if len(rot.shape) == 4:
                B, L, D1, D2 = rot.shape
                rot = rot.reshape(B * L, D1, D2)
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
                rot_6d = rot_6d.reshape(B, L, 6)
            else:
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
            signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        if self._rotation_parametrization == '6D':
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            if len(signal.shape) == 3:
                B, L, _ = signal.shape
                rot = signal[..., 3:9].reshape(B * L, 6)
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = pytorch3d_transforms.matrix_to_quaternion(mat)
                quat = quat.reshape(B, L, 4)
            else:
                rot = signal[..., 3:9]
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = pytorch3d_transforms.matrix_to_quaternion(mat)
            signal = torch.cat([signal[..., :3], quat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        goal_gripper,
        run_inference=False
    ):
        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper,
                goal_gripper
            )
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        goal_gripper = goal_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[:, :3] = self.normalize_pos(curr_gripper[:, :3])
        goal_gripper[:, :3] = self.normalize_pos(goal_gripper[:, :3])

        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(gt_trajectory)
        curr_gripper = self.convert_rot(curr_gripper)
        goal_gripper = self.convert_rot(goal_gripper)

        # Prepare inputs
        fixed_inputs = (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            goal_gripper
        )

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()

        # Add noise to the clean trajectories
        pos = self.position_noise_scheduler.add_noise(
            gt_trajectory[..., :3], noise[..., :3],
            timesteps
        )
        rot = self.rotation_noise_scheduler.add_noise(
            gt_trajectory[..., 3:9], noise[..., 3:9],
            timesteps
        )
        noisy_trajectory = torch.cat((pos, rot), -1)
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition

        # Predict the noise residual
        pred = self.policy_forward_pass(
            noisy_trajectory, timesteps,
            fixed_inputs
        )
        target = gt_trajectory

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            rot = layer_pred[..., 3:9]
            loss = (
                100 * F.l1_loss(trans, target[..., :3], reduction='mean')
                + 10 * F.l1_loss(rot, target[..., 3:9], reduction='mean')
            )
            total_loss = total_loss + loss
        return total_loss
