import einops
import torch
import torch.nn as nn

from model.utils.layers import ParallelAttention
from model.utils.encoder import Encoder
from model.utils.utils import find_traj_nn


class DiffusionHead(Encoder):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_attn_heads=8,
                 num_vis_ins_attn_layers=2,
                 num_query_cross_attn_layers=6,
                 use_instruction=False,
                 use_goal=False,
                 use_sigma=False,
                 feat_scales_to_use=1,
                 attn_rounds=1,
                 weight_tying=False,
                 rotation_parametrization='quat'):
        super().__init__(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=feat_scales_to_use,
            use_sigma=use_sigma
        )
        self.use_instruction = use_instruction
        self.use_goal = use_goal
        self.attn_rounds = attn_rounds
        self.feat_scales = feat_scales_to_use
        self.rotation_parametrization = rotation_parametrization
        if self.rotation_parametrization == '6D':
            output_dim += 2

        # Encoders
        self.traj_encoder = nn.Sequential(
            nn.Linear(9, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_encoder = nn.Linear(output_dim, embedding_dim)
        if use_goal:
            self.goal_gripper_encoder = nn.Linear(output_dim, embedding_dim)

        # Attention from vision to language
        if use_instruction and weight_tying:
            layer = ParallelAttention(
                num_layers=num_vis_ins_attn_layers,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False
            )
            self.vl_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
        elif use_instruction:
            self.vl_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=num_vis_ins_attn_layers,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=False, self_attention2=False,
                    cross_attention1=True, cross_attention2=False
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])

        # Attention from trajectory queries to language
        if weight_tying:
            layer = ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
            self.traj_lang_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
        else:
            self.traj_lang_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=1,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=False, self_attention2=False,
                    cross_attention1=True, cross_attention2=False,
                    rotary_pe=False, apply_ffn=False
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])

        # Attention from trajectory queries to context
        if weight_tying:
            layer = ParallelAttention(
                num_layers=num_query_cross_attn_layers - 2,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=True, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=True, use_adaln=True
            )
            self.traj_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
            layer = ParallelAttention(
                num_layers=2,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=True, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=True, use_adaln=True
            )
            self.pos_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
            layer = ParallelAttention(
                num_layers=2,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=True, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=True, use_adaln=True
            )
            self.rot_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
        else:
            self.traj_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=num_query_cross_attn_layers - 2,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=True, self_attention2=False,
                    cross_attention1=True, cross_attention2=False,
                    rotary_pe=True, use_adaln=True
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
            self.pos_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=2,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=True, self_attention2=False,
                    cross_attention1=True, cross_attention2=False,
                    rotary_pe=True, use_adaln=True
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
            self.rot_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=2,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=True, self_attention2=False,
                    cross_attention1=True, cross_attention2=False,
                    rotary_pe=True, use_adaln=True
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])

        # Regression after every attention to a scale
        self.pos_regressor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, 3)
            )
            for _ in range(self.attn_rounds)
            for _ in range(self.feat_scales)
        ])
        self.rot_regressor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, output_dim - 3)
            )
            for _ in range(self.attn_rounds)
            for _ in range(self.feat_scales)
        ])

    def forward(self, trajectory, trajectory_mask, timestep,
                visible_rgb, visible_pcd, curr_gripper, goal_gripper,
                instruction):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            trajectory_mask: (B, trajectory_length)
            timestep: (B, 1)
            visible_rgb: (B, num_cameras, 3, H, W) in [0, 1]
            visible_pcd: (B, num_cameras, 3, H, W) in world coordinates
            curr_gripper: (B, output_dim)
            goal_gripper: (B, output_dim)
            instruction: (B, max_instruction_length, 512)
        """
        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)
        traj_pos = self.relative_pe_layer(trajectory[..., :3])

        # Timestep features (B, 1, F)
        time_feats, time_pos = self.encode_denoising_timestep(timestep)

        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encode_images(
            visible_rgb, visible_pcd
        )

        # Encode instruction (B, 53, F)
        instr_feats, instr_pos = None, None
        if self.use_instruction:
            instr_feats, instr_pos = self.encode_instruction(instruction)

        # Encode current gripper (B, 1, F)
        curr_gripper_feats = self.curr_gripper_encoder(curr_gripper)
        curr_gripper_feats = curr_gripper_feats[:, None]
        curr_gripper_embs, curr_gripper_pos = self.encode_curr_gripper(
            curr_gripper, batch_size=len(traj_feats)
        )
        curr_gripper_feats = curr_gripper_feats + curr_gripper_embs

        # Encode goal gripper (B, 1, F)
        goal_gripper_feats, goal_gripper_pos = None, None
        if self.use_goal:
            goal_gripper_embs, goal_gripper_pos = self.encode_goal_gripper(
                goal_gripper, batch_size=len(traj_feats)
            )
            goal_gripper_feats = self.goal_gripper_encoder(goal_gripper)
            goal_gripper_feats = goal_gripper_feats[:, None]
            goal_gripper_feats = goal_gripper_feats + goal_gripper_embs

        # Attention layers
        n_trajectory = []
        for attn_round in range(self.attn_rounds):
            for scale in range(self.feat_scales):
                # Local attention
                p_inds = None
                if self.use_goal and scale > 0:
                    p_inds = find_traj_nn(
                        n_trajectory[-1][..., :3], pcd_pyramid[scale],
                        nn_=64 if scale == 1 else 16
                    )

                # One attention iteration
                update = self._one_attention_round(
                    rgb_feats_pyramid, pcd_pyramid,  # visual
                    instr_feats, instr_pos,  # language
                    curr_gripper_feats, curr_gripper_pos,  # current gripper
                    goal_gripper_feats, goal_gripper_pos,  # goal gripper
                    time_feats, time_pos,  # time
                    traj_feats, traj_pos, trajectory_mask,  # trajectory
                    attn_round, scale, p_inds
                )
                trajectory = torch.cat((
                    trajectory[..., :3] + update[..., :3],
                    update[..., 3:]
                ), -1)
                n_trajectory.append(trajectory)

        return n_trajectory

    def _one_attention_round(
        self,
        rgb_feats_pyramid, pcd_pyramid,  # visual
        instr_feats, instr_pos,  # language
        curr_gripper_feats, curr_gripper_pos,  # current gripper
        goal_gripper_feats, goal_gripper_pos,  # goal gripper
        time_feats, time_pos,  # time
        traj_feats, traj_pos, trajectory_mask,  # trajectory
        attn_round, scale, p_inds=None
    ):
        # Visual context
        context_feats = einops.rearrange(
            rgb_feats_pyramid[scale],
            "b ncam c h w -> b (ncam h w) c"
        )
        context_pos = pcd_pyramid[scale]
        if p_inds is not None:
            context_feats = torch.stack([
                f[i]  # (nn, c)
                for f, i in zip(context_feats, p_inds)
            ])
            context_pos = torch.stack([
                f[i] for f, i in zip(context_pos, p_inds)
            ])
        context_pos = self.relative_pe_layer(context_pos)

        # Language context
        if self.use_instruction:
            # Attention from vision to language
            l_offset = attn_round * self.feat_scales + scale
            context_feats, _ = self.vl_attention[l_offset](
                seq1=context_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
            )

        # Concatenate rest of context (gripper)
        context_feats = torch.cat([context_feats, curr_gripper_feats], dim=1)
        context_pos = torch.cat([context_pos, curr_gripper_pos], dim=1)

        # Concatenate goal gripper if used
        if self.use_goal:
            context_feats = torch.cat([context_feats, goal_gripper_feats], 1)
            context_pos = torch.cat([context_pos, goal_gripper_pos], 1)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        l_offset = attn_round * self.feat_scales + scale
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[l_offset](
                seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats, _ = self.traj_attention[l_offset](
            seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
            seq2=context_feats, seq2_key_padding_mask=None,
            seq1_pos=traj_pos, seq2_pos=context_pos,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None,
            ada_sgnl=time_feats.squeeze(1)
        )
        pos_feats, _ = self.pos_attention[l_offset](
            seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
            seq2=context_feats, seq2_key_padding_mask=None,
            seq1_pos=traj_pos, seq2_pos=context_pos,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None,
            ada_sgnl=time_feats.squeeze(1)
        )
        rot_feats, _ = self.rot_attention[l_offset](
            seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
            seq2=context_feats, seq2_key_padding_mask=None,
            seq1_pos=traj_pos, seq2_pos=context_pos,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None,
            ada_sgnl=time_feats.squeeze(1)
        )

        # Regress trajectory
        return torch.cat((
            self.pos_regressor[l_offset](pos_feats),
            self.rot_regressor[l_offset](rot_feats)
        ), -1)  # (B, L, output_dim)
