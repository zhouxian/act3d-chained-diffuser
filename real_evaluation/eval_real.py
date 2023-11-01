"""Real-world evaluation script on RLBench
"""
import sys
import random
from typing import Tuple, Optional
from pathlib import Path

import torch
import numpy as np
import tap
from scipy.spatial.transform import Rotation as R

from model.keypose_optimization.act3d import Baseline
from model.keypose_optimization.act3d_diffusion_model_v3 import Act3dDiffusion
from model.trajectory_optimization.diffusion_model import DiffusionPlanner
from model.trajectory_optimization.regression_model import TrajectoryRegressor
from utils.utils_without_rlbench import (
    load_episodes,
    load_instructions,
    get_gripper_loc_bounds,
)

sys.path.append('/home/zhouxian/git/franka')
from frankapy import FrankaArm
from camera.kinect import Kinect

class Arguments(tap.Tap):
    checkpoint: Path
    act3d_checkpoint: Path
    seed: int = 2
    save_img: bool = True
    device: str = "cuda"
    num_episodes: int = 1
    headless: int = 0
    max_tries: int = 10
    # max_tries: int = 30
    output: Path = Path(__file__).parent / "records.txt"
    record_actions: bool = False
    replay_actions: Optional[Path] = None
    ground_truth_rotation: bool = False
    ground_truth_position: bool = False
    ground_truth_gripper: bool = False
    task = None
    instructions: Optional[Path] = "instructions.pkl"
    arch: Optional[str] = None
    variation: int = 0
    data_dir: Path = Path(__file__).parent / "demos"
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "256,256"
    
    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "eval_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"
    
    # Toggle to switch between offline and online evaluation
    offline: int = 0

    # Toggle to switch between original HiveFormer and our models
    model: str = "baseline"  # one of "original", "baseline", "analogical"
    traj_model: str = "diffusion"  # one of "regression", "diffusion" or none

    use_rgb: int = 1
    use_goal: int = 0
    use_goal_at_test: int = 1
    dense_interpolation: int = 0
    interpolation_length: int = 100
    predict_keypose: int = 0
    predict_traj: int = 0
    exec168: int = 0

    # ---------------------------------------------------------------
    # Original HiveFormer parameters
    # ---------------------------------------------------------------

    depth: int = 4
    dim_feedforward: int = 64
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    num_layers: int = 1

    # ---------------------------------------------------------------
    # Our non-analogical baseline parameters
    # ---------------------------------------------------------------

    visualize_rgb_attn: int = 0
    gripper_loc_bounds_file: Optional[str] = None
    act3d_gripper_loc_bounds_file: Optional[str] = None
    single_task_gripper_loc_bounds: int = 0
    gripper_bounds_buffer: float = 0.01
    act3d_gripper_bounds_buffer: float = 0.01

    position_prediction_only: int = 0
    regress_position_offset: int = 0
    max_episodes: int = 0

    # Ghost points
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_ghost_points: int = 10000
    num_ghost_points_val: int = 10000

    # Model
    action_dim: int = 7
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 60
    num_ghost_point_cross_attn_layers: int = 2
    num_query_cross_attn_layers: int = 2
    act3d_num_query_cross_attn_layers: int = 2
    num_vis_ins_attn_layers: int = 2
    # one of "quat_from_top_ghost", "quat_from_query", "6D_from_top_ghost", "6D_from_query"
    rotation_parametrization: str = "quat_from_query"
    use_instruction: int = 0
    act3d_use_instruction: int = 0
    task_specific_biases: int = 0

    act3d_gripper_input_history: int = 1


def load_model(args):
    device = torch.device(args.device)

    print("Loading model from", args.checkpoint, flush=True)
    print("Loading model from", args.act3d_checkpoint, flush=True)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    diffusion_gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file,
        #task=task, buffer=0.04
        task=None, buffer=0.04
    )
    act3d_gripper_loc_bounds = get_gripper_loc_bounds(
        args.act3d_gripper_loc_bounds_file,
        task=task, buffer=args.act3d_gripper_bounds_buffer
    )

    if args.traj_model in ("diffusion", "regression"):
        if args.traj_model == "diffusion":
            diffusion_cls = DiffusionPlanner
        elif args.traj_model == "regression":
            diffusion_cls = TrajectoryRegressor

        diffusion_model = diffusion_cls(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=120,
            output_dim=7,
            num_vis_ins_attn_layers=2,
            num_query_cross_attn_layers=2,
            use_instruction=True,
            use_goal=True,
            use_goal_at_test=True,
            feat_scales_to_use=3,
            attn_rounds=1,
            weight_tying=True,
            gripper_loc_bounds=diffusion_gripper_loc_bounds,
            diffusion_head="simple"
        ).to(device)
    else:
        diffusion_model = None

    if args.model in ("act3d", "diffpose"):
        if args.model == "act3d":
            model_cls = Baseline
        elif args.model == "diffpose":
            model_cls = Act3dDiffusion

        act3d_model = model_cls(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=args.act3d_num_query_cross_attn_layers,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            rotation_parametrization=args.rotation_parametrization,
            gripper_loc_bounds=act3d_gripper_loc_bounds,
            num_ghost_points=args.num_ghost_points,
            num_ghost_points_val=args.num_ghost_points_val,
            weight_tying=bool(args.weight_tying),
            gp_emb_tying=bool(args.gp_emb_tying),
            num_sampling_level=args.num_sampling_level,
            fine_sampling_ball_diameter=args.fine_sampling_ball_diameter,
            regress_position_offset=bool(args.regress_position_offset),
            use_instruction=bool(args.act3d_use_instruction)
        ).to(device)
    else:
        act3d_model = None

    if args.predict_traj:
        diffusion_model_dict = torch.load(args.checkpoint, map_location="cpu")
        diffusion_model_dict_weight = {}
        for key in diffusion_model_dict["weight"]:
            _key = key[7:]
            diffusion_model_dict_weight[_key] = diffusion_model_dict["weight"][key]
        diffusion_model.load_state_dict(diffusion_model_dict_weight)
        diffusion_model.eval()

    if args.predict_keypose:
        act3d_model_dict = torch.load(args.act3d_checkpoint, map_location="cpu")
        act3d_model_dict_weight = {}
        for key in act3d_model_dict["weight"]:
            _key = key[7:]
            act3d_model_dict_weight[_key] = act3d_model_dict["weight"][key]
        act3d_model.load_state_dict(act3d_model_dict_weight)
        act3d_model.eval()

    return diffusion_model, act3d_model


def transform(rgb, pc):
    # normalise to [-1, 1]
    rgb = 2 * (rgb / 255.0 - 0.5) # in [-1, 1]
    rgb = rgb / 2 + 0.5  # in [0, 1]

    if rgb.shape == pc.shape == (720, 1080, 3):
        rgb = rgb[::3, ::3]
        pc = pc[::3, ::3]
    else:
        assert False

    return rgb, pc

def goto_pose(fa, pose, duration, up_offset=0.0, up_first=False, up_duration=1.5, down_offset=0.0, down_duration=1.5, down_last=False):
    if up_first:
        cur_pose = fa.get_pose()
        cur_pose.translation[2] += up_offset
        fa.goto_pose(cur_pose, duration=up_duration)

    if down_last:
        pose.translation[2] += down_offset
        fa.goto_pose(pose, duration=duration)

        pose.translation[2] -= down_offset
        fa.goto_pose(pose, duration=down_duration)
    else:
        fa.goto_pose(pose, duration=duration)

def goto_pose2(fa, pose, duration, pre_offset=np.zeros(3), pre=False, pre_duration=1.5, post_offset=np.zeros(3), post_duration=1.5, post=False):
    if pre:
        cur_pose = fa.get_pose()
        cur_pose.translation += pre_offset
        fa.goto_pose(cur_pose, duration=pre_duration)

    if post:
        pose.translation += post_offset
        fa.goto_pose(pose, duration=duration)

        pose.translation -= post_offset
        fa.goto_pose(pose, duration=post_duration)
    else:
        fa.goto_pose(pose, duration=duration)

if __name__ == "__main__":
    args = Arguments().parse_args()

    fa = FrankaArm()
    kinect = Kinect()

    print('Reset...')
    fa.reset_joints()
    print('Open gripper...')
    if args.task == 'real_reach_target':
        fa.close_gripper()
        gripper_open = False
    elif args.task == 'real_spread_sand':
        fa.open_gripper_tool()
        gripper_open = True
    else:
        fa.open_gripper()
        gripper_open = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and args
    trajectory_model, keypose_model = load_model(args)
    device = args.device

    # Evaluate
    instructions = load_instructions(args.instructions)
    max_eps_dict = load_episodes()["max_episode_length"]

    max_episodes = max_eps_dict[args.task] if args.max_episodes == 0 else args.max_episodes
    if instructions is not None:
        instr = random.choice(instructions[args.task][args.variation]).unsqueeze(0).to(device)
    else:
        instr = torch.zeros((1, 53, 512), device=device)
    task_id = torch.tensor(TASK_TO_ID[args.task]).unsqueeze(0).to(device)
    
    # for step_id in range(1):
    for step_id in range(max_episodes):
        print(step_id)
        # get obs
        rgb = kinect.get_rgb()[:, 100:-100, :]
        pcd = kinect.get_pc()[:, 100:-100, :]
        rgb, pcd = transform(rgb, pcd)
        rgb = rgb.transpose((2, 0, 1))
        pcd = pcd.transpose((2, 0, 1))
        rgbs = torch.tensor(rgb).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
        pcds = torch.tensor(pcd).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()

        gripper_pose = fa.get_pose()
        gripper_trans = gripper_pose.translation
        gripper_quat = R.from_matrix(gripper_pose.rotation).as_quat()
        gripper = np.concatenate([gripper_trans, gripper_quat, [gripper_open]])
        gripper = torch.tensor(gripper).to(device).unsqueeze(0).unsqueeze(0).float()

        padding_mask = torch.ones_like(rgbs[:, :, 0, 0, 0, 0]).bool()

        assert trajectory_model is None, f"now only support keypose estimatin"
        pred = keypose_model(
            rgb_obs=rgbs,
            pcd_obs=pcds,
            padding_mask=padding_mask,
            instruction=instr,
            gripper=gripper,
            task_id=task_id,
        )
        action = model.prepare_action(pred).detach().cpu().numpy()  # type: ignore
        print(action)
        action_gripper_open = action[0, -1] > 0.5
        # import IPython;IPython.embed()
        # move
        target_pose = fa.get_pose()
        if args.task == 'real_press_stapler' and step_id == 1:
            target_pose.translation[2] = action[0, 2] - 0.01
            target_pose.translation[2] = 0.04
        elif args.task == 'real_press_stapler' and step_id == 2:
            # target_pose.translation[2] = action[0, 2] - 0.01
            target_pose.translation[2] = 0.09
        elif args.task == 'real_press_hand_san' and step_id == 1:
            target_pose.translation = action[0, :3]
            target_pose.translation[2] -= 0.03
            import time
            time.sleep(1)
        else:
            target_pose.translation = action[0, :3]

        if args.task == 'real_spread_sand':
            pass
        else:
            target_pose.rotation = R.from_quat(action[0, 3:7]).as_matrix()

        if args.task == 'real_press_stapler' and step_id > 0:
            duration = 1.0
        elif args.task == 'real_press_hand_san' and step_id > 0:
            duration = 1.0
        else:
            duration = 5.0

        if args.task == 'real_put_fruits_in_bowl' and step_id in [0, 2]:
            goto_pose(fa, target_pose, duration=duration, down_offset=0.1, down_last=True)
        elif args.task == 'real_put_fruits_in_bowl' and step_id in [1, 3]:
            goto_pose(fa, target_pose, duration=duration, up_offset=0.16, up_first=True)
        elif args.task == 'real_stack_bowls' and step_id in [0, 2, 4]:
            target_pose.translation[2] = 0.05
            goto_pose(fa, target_pose, duration=duration, down_offset=0.05, down_duration=1.0, down_last=True)
        elif args.task == 'real_stack_bowls' and step_id in [1, 3, 5]:
            goto_pose(fa, target_pose, duration=duration, up_offset=0.10, up_duration=1.0, up_first=True, down_offset=0.04, down_duration=0.5, down_last=True)
        elif args.task == 'real_unscrew_bottle_cap' and step_id in [0]:
            target_pose.translation[2] = 0.223
            goto_pose(fa, target_pose, duration=duration, down_offset=0.05, down_duration=1.0, down_last=True)
        elif args.task == 'real_unscrew_bottle_cap' and step_id in [1]:
            target_pose.translation = fa.get_pose().translation
        elif args.task == 'real_spread_sand':
            if step_id == 0:
                target_pose.translation = np.array([0.592, 0.225, 0.135])
                goto_pose(fa, target_pose, duration=duration, down_offset=0.05, down_duration=3.0, down_last=True)
            elif step_id == 1:
                target_pose.translation[2] = 0.140
                goto_pose(fa, target_pose, duration=duration, up_offset=0.14, up_duration=4.0, up_first=True, down_offset=0.05, down_duration=1.0, down_last=True)
            elif step_id == 2:
                target_pose.translation[2] = 0.137
                goto_pose(fa, target_pose, duration=2.5)
            elif step_id == 3:
                target_pose.translation[2] = 0.137
                goto_pose(fa, target_pose, duration=4)
        elif args.task == 'real_wipe_coffee':
            if step_id == 0:
                target_pose.translation[2] = 0.005
                target_pose.translation[1] += 0.025
                target_pose.rotation = R.from_quat(np.array([ 0.99798159, -0.01401051,  0.01591191,  0.05986039])).as_matrix()
                goto_pose(fa, target_pose, duration=duration)
            elif step_id == 1:
                target_pose.translation[2] += 0.03
                goto_pose(fa, target_pose, duration=duration, up_offset=0.05, up_duration=1.0, up_first=True)
            elif step_id == 2:
                target_pose.translation[2] = 0.02
                goto_pose(fa, target_pose, duration=2.0, down_offset=0.03, down_duration=0.5, down_last=True)
            elif step_id == 3:
                target_pose.translation[2] = 0.01
                fa.goto_pose(target_pose, duration=3.5)
            else:
                goto_pose(fa, target_pose, duration=duration)
        elif args.task == 'real_put_duck_in_oven':
            if step_id == 0:
                target_pose.translation = np.array([5.6281102e-01, -0.22,  0.26118689])
                # goto_pose2(fa, target_pose, duration=duration, post_offset=np.array([0, 0.01, 0]), post=True, post_duration=1.0)
                goto_pose2(fa, target_pose, duration=duration)
            elif step_id == 2:
                # avoid collision
                temp_pose = fa.get_pose()
                temp_pose.rotation = np.array([[ 0.9600275 , -0.08617114, -0.26627778],
                       [-0.27268965, -0.50215055, -0.8206576 ],
                       [-0.06299453,  0.86046505, -0.50558604]])
                temp_pose.translation = np.array([ 0.64128851, -0.0,  0.08960278])
                goto_pose2(fa, temp_pose, duration=3.0)
                temp_pose.translation = np.array([ 0.64128851, -0.0,  0.13960278])
                goto_pose2(fa, temp_pose, duration=2.0)

                goto_pose2(fa, target_pose, duration=duration, post_offset=np.array([0, 0.0, 0.05]), post=True, post_duration=1.5)
            elif step_id == 3:
                goto_pose2(fa, target_pose, duration=duration, pre_offset=np.array([0, 0.0, 0.08]), pre=True, pre_duration=1.5)
            else:
                goto_pose(fa, target_pose, duration=duration)
        elif args.task == 'real_transfer_beans':
            if step_id == 0:
                target_pose.translation = np.array([6.5634805e-01, 0.204859,  0.138])
                goto_pose2(fa, target_pose, duration=duration)
            elif step_id == 1:
                target_pose.translation[2] = 0.165
                # goto_pose2(fa, target_pose, duration=duration, pre_offset=np.array([0, 0.0, 0.20]), pre=True, pre_duration=1.5, post_offset=np.array([0, 0.0, 0.06]), post=True, post_duration=0.5)
                goto_pose2(fa, target_pose, duration=duration, pre_offset=np.array([0, 0.0, 0.20]), pre=True, pre_duration=3.0, post_offset=np.array([0, 0.0, 0.02]), post=True, post_duration=1.0)
            elif step_id == 2:
                target_pose.translation[2] = 0.13
                goto_pose(fa, target_pose, duration=duration)
            elif step_id == 3:
                target_pose.translation[2] = 0.13
                goto_pose(fa, target_pose, duration=duration)
            elif step_id == 4:
                temp_pose = fa.get_pose()
                temp_pose.translation = target_pose.translation
                temp_pose.translation[1] += 0.04
                temp_pose.translation[2] += 0.03
                target_pose.translation[2] += 0.02
                goto_pose(fa, temp_pose, duration=4.0)
                goto_pose(fa, target_pose, duration=2.0)
            else:
                goto_pose(fa, target_pose, duration=duration)
        else:
            fa.goto_pose(target_pose, duration=duration)

        if gripper_open and not action_gripper_open:
            if args.task == 'real_unscrew_bottle_cap':
                fa.close_gripper_soft()
            else:
                fa.close_gripper()
            gripper_open = False
        elif not gripper_open and action_gripper_open:
            fa.open_gripper()
            gripper_open = True


    if args.task == 'real_put_duck_in_oven':
        pose = fa.get_pose()
        pose.translation[1] += 0.06
        goto_pose(fa, pose, duration=1.5)

    fa.reset_joints()
    fa.open_gripper()




