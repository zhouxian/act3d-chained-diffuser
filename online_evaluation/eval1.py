"""Online evaluation script on RLBench."""
import random
from typing import Tuple, Optional
from pathlib import Path
import json
import os

import torch
import numpy as np
import tap

from model.keypose_optimization.act3d import Act3D
from model.trajectory_optimization.diffusion_model import DiffusionPlanner
from utils.utils_without_rlbench import (
    load_episodes,
    load_instructions,
    get_gripper_loc_bounds,
    round_floats
)
from online_evaluation.utils_with_rlbench import RLBenchEnv, Actioner


class Arguments(tap.Tap):
    diff_checkpoint: Path
    act3d_checkpoint: Path
    seed: int = 2
    save_img: bool = True
    device: str = "cuda"
    num_episodes: int = 1
    headless: int = 0
    max_tries: int = 10
    tasks: Optional[Tuple[str, ...]] = None
    instructions: Optional[Path] = "instructions.pkl"
    arch: Optional[str] = None
    variations: Tuple[int, ...] = (-1,)
    data_dir: Path = Path(__file__).parent / "demos"
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "256,256"
    verbose: int = 0
    output_file: Path = Path(__file__).parent / "eval.json"
    
    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "eval_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"
    
    # Toggle to switch between offline and online evaluation
    # 0: use predicted keypose, 1: use ground-truth keypose
    offline: int = 0

    # Toggle to switch between original HiveFormer and our models
    model: str = "act3d"  # one of "act3d", "diffpose"
    traj_model: str = "diffusion"  # one of "regression", "diffusion"

    max_steps: int = 50
    collision_checking: int = 0
    dense_interpolation: int = 0
    interpolation_length: int = 100
    predict_keypose: int = 0
    predict_traj: int = 0

    gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    act3d_gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    single_task_gripper_loc_bounds: int = 0

    # Model
    action_dim: int = 7
    use_instruction: int = 1
    act3d_use_instruction: int = 0


def load_models(args):
    device = torch.device(args.device)

    print("Loading model from", args.diff_checkpoint, flush=True)
    print("Loading model from", args.act3d_checkpoint, flush=True)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    print('Trajectory gripper workspace')
    diffusion_gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file,
        task=None, buffer=0.04
    )
    print('Keypose gripper workspace')
    act3d_gripper_loc_bounds = get_gripper_loc_bounds(
        args.act3d_gripper_loc_bounds_file,
        task=task, buffer=0.04
    )

    if args.predict_traj:
        diffusion_model = DiffusionPlanner(
            backbone="clip",
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=120,
            output_dim=7,
            num_vis_ins_attn_layers=2,
            num_query_cross_attn_layers=6,
            use_instruction=True,
            use_goal=True,
            use_goal_at_test=False,
            feat_scales_to_use=1,
            attn_rounds=1,
            weight_tying=True,
            gripper_loc_bounds=diffusion_gripper_loc_bounds,
            rotation_parametrization='6D',
            diffusion_timesteps=100
        ).to(device)
    else:
        diffusion_model = None

    if args.predict_keypose:
        act3d_model = Act3D(
            backbone="clip",
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=60,
            num_ghost_point_cross_attn_layers=2,
            num_query_cross_attn_layers=2,
            num_vis_ins_attn_layers=2,
            rotation_parametrization="quat_from_query",
            gripper_loc_bounds=act3d_gripper_loc_bounds,
            num_ghost_points=10000,
            num_ghost_points_val=10000,
            weight_tying=True,
            gp_emb_tying=False,
            num_sampling_level=3,
            fine_sampling_ball_diameter=0.16,
            regress_position_offset=False,
            use_instruction=bool(args.act3d_use_instruction)
        ).to(device)
    else:
        act3d_model = None

    if args.predict_traj:
        diffusion_model_dict = torch.load(args.diff_checkpoint, map_location="cpu")
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


if __name__ == "__main__":
    # Arguments
    args = Arguments().parse_args()
    args.cameras = tuple(x for y in args.cameras for x in y.split(","))
    print("Arguments:")
    print(args)
    print("-" * 100)
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    traj_model, keypose_model = load_models(args)

    # Load RLBench environment
    env = RLBenchEnv(
        data_path=args.data_dir,
        traj_cmd=args.predict_keypose,
        exec168=False,
        image_size=[int(x) for x in args.image_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        headless=bool(args.headless),
        apply_cameras=args.cameras,
        collision_checking=bool(args.collision_checking)
    )

    instruction = load_instructions(args.instructions)
    if instruction is None:
        raise NotImplementedError()

    actioner = Actioner(
        keypose_model=keypose_model,
        traj_model=traj_model,
        instructions=instruction,
        apply_cameras=args.cameras,
        action_dim=args.action_dim,
        predict_keypose=bool(args.predict_keypose),
        predict_trajectory=bool(args.predict_traj)
    )
    max_eps_dict = load_episodes()["max_episode_length"]
    task_success_rates = {}

    for task_str in args.tasks:
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=(
                max_eps_dict[task_str] if args.max_steps == -1
                else args.max_steps
            ),
            num_variations=args.variations[-1] + 1,
            num_demos=args.num_episodes,
            actioner=actioner,
            log_dir=log_dir / task_str if args.save_img else None,
            max_tries=args.max_tries,
            save_attn=False,
            dense_interpolation=bool(args.dense_interpolation),
            interpolation_length=args.interpolation_length,
            record_videos=False,
            position_prediction_only=False,
            offline=args.offline,
            verbose=bool(args.verbose)
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
