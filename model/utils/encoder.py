import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork

from .position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from .resnet import load_resnet50
from .clip import load_clip


class Encoder(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_sampling_level=3,
                 use_sigma=False):
        super().__init__()
        assert backbone in ["resnet", "clip"]
        assert image_size in [(128, 128), (256, 256)]
        assert num_sampling_level in [1, 2, 3, 4]

        self.image_size = image_size
        self.num_sampling_level = num_sampling_level

        # Frozen backbone
        if backbone == "resnet":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        self.feature_pyramid = FeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048], embedding_dim
        )
        if self.image_size == (128, 128):
            # Coarse RGB features are the 2nd layer of the feature pyramid
            # at 1/4 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (64x64)
            self.coarse_feature_map = ['res2', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == (256, 256):
            # Coarse RGB features are the 3rd layer of the feature pyramid
            # at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (128x128)
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [8, 2, 2, 2]

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(1, embedding_dim)

        # Goal gripper learnable features
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)

        # Instruction encoder
        self.instruction_encoder = nn.Linear(512, embedding_dim)

        # Time embeddings
        if not use_sigma:
            self.time_emb = SinusoidalPosEmb(embedding_dim)
        else:
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            )

    def forward(self):
        return None

    def encode_curr_gripper(self, curr_gripper, batch_size=1):
        """
        Compute current gripper position features and positional embeddings.

        Args:
            - curr_gripper: (B, 3+)

        Returns:
            - curr_gripper_feats: (B, 1, F)
            - curr_gripper_pos: (B, 1, F, 2)
        """
        curr_gripper_feats = self.curr_gripper_embed.weight.repeat(
            batch_size, 1
        ).unsqueeze(1)
        curr_gripper_pos = self.relative_pe_layer(curr_gripper[:, :3][:, None])
        return curr_gripper_feats, curr_gripper_pos

    def encode_goal_gripper(self, goal_gripper, batch_size=1):
        """
        Compute goal gripper position features and positional embeddings.

        Args:
            - goal_gripper: (B, 3+)

        Returns:
            - goal_gripper_feats: (B, 1, F)
            - goal_gripper_pos: (B, 1, F, 2)
        """
        goal_gripper_feats = self.goal_gripper_embed.weight.repeat(
            batch_size, 1
        ).unsqueeze(1)
        goal_gripper_pos = self.relative_pe_layer(goal_gripper[:, :3][:, None])
        return goal_gripper_feats, goal_gripper_pos

    def encode_images(self, rgb, pcd):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        num_cameras = rgb.shape[1]

        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)

        # Pass visual features through feature pyramid network
        rgb_features = self.feature_pyramid(rgb_features)

        # Treat different cameras separately
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        rgb_feats_pyramid = []
        pcd_pyramid = []
        for i in range(self.num_sampling_level):
            # Isolate level's visual features
            rgb_features_i = rgb_features[self.feature_map_pyramid[i]]

            # Interpolate xy-depth to get the locations for this level
            pcd_i = F.interpolate(
                pcd,
                scale_factor=1. / self.downscaling_factor_pyramid[i],
                mode='bilinear'
            )

            # Merge different cameras for clouds, separate for rgb features
            h, w = pcd_i.shape[-2:]
            pcd_i = einops.rearrange(
                pcd_i,
                "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )
            rgb_features_i = einops.rearrange(
                rgb_features_i,
                "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
            )

            rgb_feats_pyramid.append(rgb_features_i)
            pcd_pyramid.append(pcd_i)

        return rgb_feats_pyramid, pcd_pyramid

    def encode_instruction(self, instruction):
        """
        Compute language features/pos embeddings on top of CLIP features.

        Args:
            - instruction: (B, max_instruction_length, 512)

        Returns:
            - instr_feats: (B, 53, F)
            - instr_dummy_pos: (B, 53, F, 2)
        """
        instr_feats = self.instruction_encoder(instruction)
        # Dummy positional embeddings, all 0s
        instr_dummy_pos = torch.zeros(
            len(instruction), instr_feats.shape[1], 3,
            device=instruction.device
        )
        instr_dummy_pos = self.relative_pe_layer(instr_dummy_pos)
        return instr_feats, instr_dummy_pos

    def encode_denoising_timestep(self, timestep):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, 1, F)
            - time_pos: (B, 1, F, 2)
        """
        time_feats = self.time_emb(timestep).unsqueeze(1)  # (B, 1, F)
        time_pos = torch.zeros(len(timestep), 1, 3, device=timestep.device)
        time_pos = self.relative_pe_layer(time_pos)
        return time_feats, time_pos
