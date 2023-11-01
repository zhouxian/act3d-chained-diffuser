import random
import itertools
from typing import Tuple, Dict, List
from pathlib import Path
import json
from tqdm import tqdm
import tap
import torch
import numpy as np
import einops
from rlbench.demo import Demo
from utils.utils_with_rlbench import (
    RLBenchEnv,
    keypoint_discovery,
    obs_to_attn,
    transform,
)


class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent / "c2farm"
    seed: int = 2
    tasks: Tuple[str, ...] = ("stack_wine",)
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "128,128"
    output: Path = Path(__file__).parent / "datasets"
    max_variations: int = 1
    offset: int = 0
    num_workers: int = 0


def get_attn_indices_from_demo(
    task_str: str, demo: Demo, cameras: Tuple[str, ...]
) -> List[Dict[str, Tuple[int, int]]]:
    frames = keypoint_discovery(demo)

    # HACK tower3
    if task_str == "tower3":
        frames = [k for i, k in enumerate(frames) if i % 6 in set([1, 4])]

    # HACK tower4
    elif task_str == "tower4":
        frames = frames[6:]

    frames.insert(0, 0)
    return [{cam: obs_to_attn(demo[f], cam) for cam in cameras} for f in frames]


def get_observation(task_str: str, variation: int, episode: int, env: RLBenchEnv):
    demos = env.get_demo(task_str, variation, episode)
    demo = demos[0]

    key_frame = keypoint_discovery(demo)
    # HACK for tower3
    if task_str == "tower3":
        key_frame = [k for i, k in enumerate(key_frame) if i % 6 in set([1, 4])]
    # HACK tower4
    elif task_str == "tower4":
        key_frame = key_frame[6:]
    key_frame.insert(0, 0)

    state_ls = []
    action_ls = []
    for f in key_frame:
        state, action = env.get_obs_action(demo._observations[f])
        state = transform(state)
        state_ls.append(state.unsqueeze(0))
        action_ls.append(action.unsqueeze(0))

    return demo, state_ls, action_ls


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: Arguments):
        # load RLBench environment
        self.env = RLBenchEnv(
            data_path=args.data_dir,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            apply_cameras=args.cameras,
        )

        with open("data_preprocessing/episodes.json") as fid:
            episodes = json.load(fid)
        self.max_eps_dict = episodes["max_episode_length"]
        self.variable_lengths = set(episodes["variable_length"])

        for task_str in args.tasks:
            if task_str in self.max_eps_dict:
                continue
            try:
                _, state_ls, _ = get_observation(task_str, args.offset, 0, self.env)
            except:
                print(f"Invalid demo for {task_str}")
                continue
            self.max_eps_dict[task_str] = len(state_ls) - 1
            raise ValueError(
                f"Guessing that the size of {task_str} is {len(state_ls) - 1}"
            )

        broken = set(episodes["broken"])
        tasks = [t for t in args.tasks if t not in broken]
        variations = range(args.offset, args.max_variations)
        self.items = []
        for task_str, variation in itertools.product(tasks, variations):
            episodes_dir = args.data_dir / task_str / f"variation{variation}" / "episodes"
            episodes = [
                (task_str, variation, int(ep.stem[7:]))
                for ep in episodes_dir.glob("episode*")
            ]
            self.items += episodes

        self.num_items = len(self.items)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]
        taskvar_dir = args.output / f"{task}+{variation}"
        taskvar_dir.mkdir(parents=True, exist_ok=True)

        try:
            demo, state_ls, action_ls = get_observation(
                task, variation, episode, self.env
            )
        except (FileNotFoundError, RuntimeError, IndexError, EOFError) as e:
            print(e)
            return

        state_ls = einops.rearrange(
            state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=len(args.cameras),
            m=2,
        )

        frame_ids = list(range(len(state_ls) - 1))
        num_frames = len(frame_ids)
        attn_indices = get_attn_indices_from_demo(task, demo, args.cameras)

        if (task in self.variable_lengths and num_frames > self.max_eps_dict[task]) or (
            task not in self.variable_lengths and num_frames != self.max_eps_dict[task]
        ):
            print(f"ERROR ({task}, {variation}, {episode})")
            print(f"\t {len(frame_ids)} != {self.max_eps_dict[task]}")
            return

        state_dict: List = [[] for _ in range(5)]
        print("Demo {}".format(episode))
        state_dict[0].extend(frame_ids)
        state_dict[1].extend(state_ls[:-1])
        state_dict[2].extend(action_ls[1:])
        state_dict[3].extend(attn_indices)
        state_dict[4].extend(action_ls[:-1])  # gripper pos

        np.save(taskvar_dir / f"ep{episode}.npy", state_dict)  # type: ignore


if __name__ == "__main__":
    args = Arguments().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = Dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    for _ in tqdm(dataloader):
        continue
