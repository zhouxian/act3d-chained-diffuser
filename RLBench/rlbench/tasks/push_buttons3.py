from typing import List, Tuple
import itertools
import random
import math
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import JointCondition, ConditionSet


Color = Tuple[str, Tuple[float, float, float]]

class PushButtons3(Task):
    num_buttons = 3
    max_variations = 5220
    # button top plate and wrapper will be be red before task completion
    # and be changed to cyan upon success of task, so colors list used to randomly vary colors of
    # base block will be redefined, excluding red and green
    colors: List[Color] = [
        ("maroon", (0.5, 0.0, 0.0)),
        ("green", (0.0, 0.5, 0.0)),
        ("blue", (0.0, 0.0, 1.0)),
        ("navy", (0.0, 0.0, 0.5)),
        ("yellow", (1.0, 1.0, 0.0)),
        ("cyan", (0.0, 1.0, 1.0)),
        ("magenta", (1.0, 0.0, 1.0)),
        ("silver", (0.75, 0.75, 0.75)),
        ("gray", (0.5, 0.5, 0.5)),
        ("orange", (1.0, 0.5, 0.0)),
        ("olive", (0.5, 0.5, 0.0)),
        ("purple", (0.5, 0.0, 0.5)),
        ("teal", (0, 0.5, 0.5)),
        ("azure", (0.0, 0.5, 1.0)),
        ("violet", (0.5, 0.0, 1.0)),
        ("rose", (1.0, 0.0, 0.5)),
        ("black", (0.0, 0.0, 0.0)),
        ("white", (1.0, 1.0, 1.0)),
    ]
    _sequences = None

    def init_task(self) -> None:
        self.buttons_pushed = 0
        self.color_variation_index = 0
        self.target_buttons = [
            Shape("push_buttons_target%d" % i) for i in range(self.num_buttons)
        ]
        self.target_topPlates = [
            Shape("target_button_topPlate%d" % i) for i in range(self.num_buttons)
        ]
        self.target_joints = [
            Joint("target_button_joint%d" % i) for i in range(self.num_buttons)
        ]
        self.target_wraps = [
            Shape("target_button_wrap%d" % i) for i in range(self.num_buttons)
        ]
        self.boundaries = Shape("push_buttons_boundary")
        # goal_conditions merely state joint conditions for push action for
        # each button regardless of whether the task involves pushing it
        self.goal_conditions = [
            JointCondition(self.target_joints[n], 0.003)
            for n in range(self.num_buttons)
        ]

        self.register_waypoint_ability_start(0, self._move_above_next_target)
        self.register_waypoints_should_repeat(self._repeat)

    @property
    def sequences(self) -> List[Tuple[Color, ...]]:
        if self._sequences is None:
            sequences = [set(), set(), set()]
            for col in itertools.permutations(self.colors, self.num_buttons):
                for i in range(3):
                    seq = tuple(col[: i + 1])
                    sequences[i].add(seq)
            var_rand = random.Random(3)
            self._sequences = []
            for seq in sequences:
                seq2 = sorted(seq)
                var_rand.shuffle(seq2)
                self._sequences += seq2[: self.max_variations]
            var_rand.shuffle(self._sequences)
            self._sequences = self._sequences[: self.max_variations]
        return self._sequences

    def init_episode(self, index: int) -> List[str]:
        for b  in self.target_buttons:
            b.set_color([0.9, 0.9, 0.9])

        # For each color permutation, we want to have 1, 2 or 3 buttons pushed
        button_colors = self.sequences[index]
        self.buttons_to_push = len(button_colors)

        self.color_names = []
        self.color_rgbs = []
        self.chosen_colors = []
        for (color_name, color_rgb), tp, w in zip(button_colors, self.target_topPlates, self.target_wraps):
            self.color_names.append(color_name)
            self.color_rgbs.append(color_rgb)
            self.chosen_colors.append((color_name, color_rgb))
            w.set_color(list(color_rgb))
            tp.set_color(list(color_rgb))

        # for task success, all button to push must have green color RGB
        self.success_conditions = []
        for i in range(self.buttons_to_push):
            self.success_conditions.append(self.goal_conditions[i])

        self.register_success_conditions([
            ConditionSet(
                self.success_conditions,
                order_matters=True,
                simultaneously_met=False
                )
            ]
        )

        rtn0 = "push the %s button" % self.color_names[0]
        rtn1 = "press the %s button" % self.color_names[0]
        rtn2 = "push down the button with the %s color" % self.color_names[0]
        for i in range(self.buttons_to_push):
            if i == 0:
                continue
            else:
                rtn0 += ", then push the %s button" % self.color_names[i]
                rtn1 += ", then press the %s button" % self.color_names[i]
                rtn2 += ", then the %s one" % self.color_names[i]

        b = SpawnBoundary([self.boundaries])
        for button in self.target_buttons:
            b.sample(button, min_distance=0.1)

        num_non_targets = 3 - self.buttons_to_push
        spare_colors = list(
            set(self.colors)
            - set([self.chosen_colors[i] for i in range(self.buttons_to_push)])
        )

        spare_color_rgbs = []
        for i in range(len(spare_colors)):
            _, rgb = spare_colors[i]
            spare_color_rgbs.append(rgb)

        color_choice_indexes = np.random.choice(
            range(len(spare_colors)), size=num_non_targets, replace=False
        )
        non_target_index = 0
        for i, (tp, w) in enumerate(zip(self.target_topPlates, self.target_wraps)):
            if i in range(self.buttons_to_push):
                pass
            else:
                _, rgb = spare_colors[color_choice_indexes[non_target_index]]
                tp.set_color(list(rgb))
                w.set_color(list(rgb))
                non_target_index += 1

        return [rtn0, rtn1, rtn2]

    def variation_count(self) -> int:
        return np.minimum(len(self.sequences), self.max_variations)

    def step(self) -> None:
        for i in range(len(self.target_buttons)):
            if self.goal_conditions[i].condition_met() == (True, True):
                self.target_topPlates[i].set_color([0.0, 1.0, 0.0])
                self.target_wraps[i].set_color([0.0, 1.0, 0.0])

    def cleanup(self) -> None:
        self.buttons_pushed = 0

    def _move_above_next_target(self, waypoint):
        if self.buttons_pushed >= self.buttons_to_push:
            print(
                "buttons_pushed:",
                self.buttons_pushed,
                "buttons_to_push:",
                self.buttons_to_push,
            )
            raise RuntimeError("Should not be here.")
        w0 = Dummy("waypoint0")
        x, y, z = self.target_buttons[self.buttons_pushed].get_position()
        w0.set_position([x, y, z + 0.083])
        w0.set_orientation([math.pi, 0, math.pi])

    def _repeat(self):
        self.buttons_pushed += 1
        return self.buttons_pushed < self.buttons_to_push
