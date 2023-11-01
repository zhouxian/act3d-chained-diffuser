from typing import List, Tuple, Optional, Set
import itertools
import random
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.object import Object
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, Condition
from rlbench.backend.conditions import NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors, state_size, shape_size


Color = Tuple[str, Tuple[float, float, float]]


class Tower3(Task):
    max_variations = 200
    num_target_blocks = 3
    _sequences: Optional[List[Tuple[Color, ...]]] = None

    @property
    def sequences(self) -> List[Tuple[Color, ...]]:
        if self._sequences is None:
            var_rand = random.Random(3)
            self._sequences = []

            sequences: List[Set[Tuple[Color, ...]]] = [
                set() for _ in range(self.num_target_blocks)
            ]

            for col in itertools.permutations(colors, self.num_target_blocks):
                for i in range(self.num_target_blocks):
                    block_colors = tuple(col[: i + 1])
                    sequences[i].add(block_colors)

            for seq in sequences:
                seq2 = sorted(list(seq))
                var_rand.shuffle(seq2)
                self._sequences += seq2[: self.max_variations]
            var_rand.shuffle(self._sequences)
            self._sequences = self._sequences[: self.max_variations]

        return self._sequences

    def init_task(self) -> None:
        assert len(colors) >= self.num_target_blocks

        self.blocks_stacked = 0
        self.target_blocks = [
            Shape("stack_blocks_target%d" % i) for i in range(self.num_target_blocks)
        ]

        self.boundaries = [Shape("stack_blocks_boundary%d" % i) for i in range(4)]

        self.register_graspable_objects(self.target_blocks)

        self.register_waypoint_ability_start(0, self._move_above_next_target)
        self.register_waypoint_ability_start(3, self._move_above_drop_zone)
        self.register_waypoint_ability_start(5, self._is_last)
        self.register_waypoints_should_repeat(self._repeat)

    def init_episode(self, index: int) -> List[str]:
        # Colorize blocks from the tower
        self.block_colors = self.sequences[index]
        self.blocks_to_stack = len(self.block_colors)
        for (color_name, color_rgb), b in zip(self.block_colors, self.target_blocks):
            b.set_color(list(color_rgb))

        success_detector = ProximitySensor("stack_blocks_success")
        success_conditions: List[Condition] = [
            DetectedCondition(b, success_detector)
            for b in self.target_blocks[: self.blocks_to_stack]
        ]
        self.register_success_conditions(
            [
                ConditionSet(
                    success_conditions, order_matters=True, simultaneously_met=False
                ),
                NothingGrasped(self.robot.gripper),
            ]
        )

        self.blocks_stacked = 0

        # Colorize other blocks with other colors
        num_non_targets = self.num_target_blocks - self.blocks_to_stack
        spare_colors = [c for c in colors if c not in self.block_colors]
        color_choice_indexes = np.random.choice(
            range(len(spare_colors)), size=num_non_targets, replace=False
        )
        non_target_index = 0
        spare_blocks = self.target_blocks[self.blocks_to_stack :]
        for block in spare_blocks:
            _, color_rgb = spare_colors[color_choice_indexes[non_target_index]]
            block.set_color(list(color_rgb))
            non_target_index += 1

        b = SpawnBoundary(self.boundaries)
        for block in self.target_blocks:
            b.sample(block, min_distance=0.1)

        # step by step
        instructions = []
        for _ in range(5):
            instr = ""
            for i, (color, _) in enumerate(self.block_colors):
                if i == 0:
                    instr = random.choice(
                        [
                            f"stack the {color} block",
                            f"first place the {color} block",
                            f"pick up and set down the {color} block",
                        ]
                    )
                else:
                    instr += random.choice([". Then ", ", then ", " ", " and "])
                    prev, _ = self.block_colors[i - 1]
                    instr += random.choice(
                        [
                            f"stack the {color} block on top of it",
                            f"place the {color} block on top of the {prev} one",
                            f"add the {color} cube",
                        ]
                    )
            instructions.append(instr)

        # enumeration
        color_names = [c for c, _ in self.block_colors]
        ordered_colors = ", ".join(color_names)
        instructions += [
            f"Stack the {ordered_colors} blocks.",
            f"Place the {ordered_colors} cubes on top of each other.",
            f"Pick up and set down {ordered_colors} blocks on top of each other.",
            f"Build a tall tower out of {ordered_colors} cubes.",
            f"Arrange the {ordered_colors} blocks in a vertical stack on the table top.",
            f"Set {ordered_colors} cubes on top of each other.",
        ]

        return instructions

    def variation_count(self) -> int:
        return self.max_variations

    def _move_above_next_target(self, _):
        if self.blocks_stacked >= self.blocks_to_stack:
            raise RuntimeError("Should not be here.")
        w2 = Dummy("waypoint1")
        x, y, z = self.target_blocks[self.blocks_stacked].get_position()
        _, _, oz = self.target_blocks[self.blocks_stacked].get_orientation()
        ox, oy, _ = w2.get_orientation()
        w2.set_position([x, y, z])
        w2.set_orientation([ox, oy, -oz])

    def _move_above_drop_zone(self, waypoint):
        target = Shape("stack_blocks_target_plane")
        x, y, z = target.get_position()
        waypoint.get_waypoint_object().set_position(
            [x, y, z + 0.08 + 0.06 * self.blocks_stacked]
        )

    def _is_last(self, waypoint):
        last = self.blocks_stacked == self.blocks_to_stack - 1
        waypoint.skip = last

    def _repeat(self):
        self.blocks_stacked += 1
        return self.blocks_stacked < self.blocks_to_stack

    @property
    def state(self) -> np.ndarray:
        """
        Return a vector containing information for all objects in the scene
        """
        if not hasattr(self, "target_blocks"):
            raise RuntimeError("Please initialize the task first")

        # sort objects according to their x coord
        shapes = sorted(self.target_blocks, key=_get_color)

        info = np.concatenate([_get_shape_pose(shape) for shape in shapes])

        state = np.zeros(state_size)
        state[: info.size] = info

        return state


def _get_color(shape: Shape) -> List[float]:
    return list(shape.get_color())


def _get_shape_pose(shape: Object) -> np.ndarray:
    shape_state = np.concatenate([shape.get_position(), shape.get_quaternion()])
    pad_length = shape_size - shape_state.size
    assert pad_length >= 0
    return np.pad(shape_state, (0, pad_length))
