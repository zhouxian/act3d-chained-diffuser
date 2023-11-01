import os
import pickle
import glob

import numpy as np
import blosc
import torch
import PIL.Image as Image
import moviepy


from model.keypose_optimization.act3d_diffusion_utils import (
    visualize_actions_and_point_clouds
)

def visualize_actions_and_point_clouds_video(visible_pcd, visible_rgb,
                                             gt_pose, curr_pose,
                                             save=True, rotation_param="quat_from_query"):
    """Visualize by plotting the point clouds and gripper pose as video.

    Args:
        visible_pcd: A tensor of shape (B, ncam, 3, H, W)
        visible_rgb: A tensor of shape (B, ncam, 3, H, W)
        gt_pose: A tensor of shape (B, 8)
        curr_pose: A tensor of shape (B, 8)
    """
    images, rand_inds = [], None
    for i in range(visible_pcd.shape[0]):
        # `visualize_actions_and_point_clouds` only visualize the first
        # point cloud and gripper in the batch.
        # To overlap two scenes in the same visualization, you can
        # 1) concatenate one scene to another at the `width` dimension.
        #    visible_pcd, visible_rgb becomes (B, ncam, 3, H, W * 2)
        # 2) for the gt_pose and curr_pose argument in the function,
        #    you can set the former to the gt_action of the first scene
        #    and the later to the gt_action of the second scene
        image, rand_inds = visualize_actions_and_point_clouds(
            visible_pcd[i:], visible_rgb[i:],
            [gt_pose[i:], curr_pose[i:]], 
            ["gt", "curr"], # add legened label to the imagined gripper
            ["d", "o"], # some dummy matplotlib marker
            save=False,
            rotation_param=rotation_param,
            rand_inds=rand_inds,
        )
        # add denoising progress bar
        images.append(image)
    pil_images = []
    for img in images:
        pil_images.extend([Image.fromarray(img)] * 2)
    pil_images[0].save("keypose_frames.gif", save_all=True,
                        append_images=pil_images[1:], duration=1, loop=0)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=1)
    clip.write_videofile("keypose_frames.mp4")

def load_pkl(data_dir):
    file_paths = glob.glob(os.path.join(data_dir, "*.pkl"))
    file_path = file_paths[np.random.randint(len(file_paths))]
    content = pickle.loads(open(file_path, "rb").read())
    rgb = np.stack([content[1][i][:, 0] for i in range(len(content[0]))], axis=0)
    pcd = np.stack([content[1][i][:, 1] for i in range(len(content[0]))], axis=0)
    gt_action = np.concatenate([content[2][i] for i in range(len(content[0]))], axis=0)
    curr_gripper = np.concatenate([content[4][i] for i in range(len(content[0]))], axis=0)

    rgb = (torch.as_tensor(rgb).float() + 1) / 2
    pcd = torch.as_tensor(pcd).float()
    gt_action = torch.as_tensor(gt_action).float()
    curr_gripper = torch.as_tensor(curr_gripper).float()

    return rgb, pcd, gt_action, curr_gripper

def main():
    rgb, pcd, gt_action, curr_gripper = load_pkl('/projects/katefgroup/robomimic/lift/ph/')
    visualize_actions_and_point_clouds_video(pcd, rgb, gt_action, curr_gripper)

if __name__ == "__main__":
    main()
