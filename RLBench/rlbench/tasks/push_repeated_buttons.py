from typing import List, Tuple, Optional
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


class PushRepeatedButtons(Task):
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
        self.goal_conditions = []

        self.register_waypoint_ability_start(0, self._move_above_next_target)
        self.register_waypoints_should_repeat(self._repeat)

    @property
    def sequences(self) -> List[Tuple[Tuple[Optional[int], Optional[int]], Tuple[Color, ...]]]:
        if self._sequences is None:
            sequences_per_button = [set(), set(), set()]
            for col in itertools.permutations(self.colors, self.num_buttons):
                for i in range(3):
                    seq = tuple(col[: i + 1])
                    sequences_per_button[i].add(seq)
            var_rand = random.Random(3)
            sequences = []
            for seq in sequences_per_button:
                seq2 = sorted(seq)
                var_rand.shuffle(seq2)
                sequences += seq2[: self.max_variations]
            var_rand.shuffle(sequences)
            sequences = sequences[: self.max_variations]

            var_rand = random.Random(0)
            self._sequences = []
            # augment variations with duplicated steps
            for var in sequences:
                # it doesn't work 1 single button since we dont have the final image
                if len(var) == 1:
                    self._sequences.append(((None, None), var))
                    continue
                # it should not be systematic!
                if var_rand.random() > 0.5:
                    self._sequences.append(((None, None), var))
                    continue
                orig_step = var_rand.randint(0, len(var) - 2)
                new_step = var_rand.randint(0, len(var) - 1)
                new_var = list(var).copy()
                new_var.insert(new_step, var[orig_step])
                self._sequences.append(((orig_step, new_step), new_var))
            self._sequences = self._sequences[: self.max_variations]
        return self._sequences

    def init_episode(self, index: int) -> List[str]:
        for b in self.target_buttons:
            b.set_color([0.5, 0.5, 0.5])

        # For each color permutation, we want to have 1, 2 or 3 buttons pushed
        (orig_step, new_step), button_colors = self.sequences[index]
        num_targets = len(set(button_colors))
        self.button_indices = list(range(num_targets))
        if orig_step is not None and new_step is not None:
            self.button_indices.insert(new_step, self.button_indices[orig_step])

        self.buttons_to_push = len(self.button_indices)

        self.color_names = {}
        self.color_rgbs = {}
        self.chosen_colors = {}
        for i, index in enumerate(self.button_indices):
            color_name, color_rgb = button_colors[i]

            tp = self.target_topPlates[index]
            tp.set_color(list(color_rgb))

            w = self.target_wraps[index]
            w.set_color(list(color_rgb))

            self.color_names[index] = color_name
            self.color_rgbs[index] = color_rgb
            self.chosen_colors[index] = (color_name, color_rgb)

        # for task success, all button to push must have green color RGB
        self.success_conditions = []
        self.goal_conditions = []
        for i, index in enumerate(self.button_indices):
            self.goal_conditions.append(JointCondition(self.target_joints[index], 0.003))
            self.success_conditions.append(self.goal_conditions[i])

        self.register_success_conditions(
            [ConditionSet(self.success_conditions, True, False)]
        )

        rtn0 = "push the %s button" % self.color_names[self.button_indices[0]]
        rtn1 = "press the %s button" % self.color_names[self.button_indices[0]]
        rtn2 = "push down the button with the %s base" % self.color_names[self.button_indices[0]]
        for i, index in enumerate(self.button_indices):
            if i == 0:
                continue
            else:
                rtn0 += ", then push the %s button" % self.color_names[index]
                rtn1 += ", then press the %s button" % self.color_names[index]
                rtn2 += ", then the %s one" % self.color_names[index]

        b = SpawnBoundary([self.boundaries])
        for button in self.target_buttons:
            b.sample(button, min_distance=0.1)

        num_non_targets = 3 - num_targets
        spare_colors = list(
            set(self.colors) - set([self.chosen_colors[i] for i in range(num_targets)])
        )

        spare_color_rgbs = []
        for i in range(len(spare_colors)):
            _, rgb = spare_colors[i]
            spare_color_rgbs.append(rgb)

        color_choice_indexes = np.random.choice(
            range(len(spare_colors)), size=num_non_targets, replace=False
        )
        non_target_index = 0
        for i, button in enumerate(self.target_buttons):
            if i < num_targets:
                pass
            else:
                _, rgb = spare_colors[color_choice_indexes[non_target_index]]
                button.set_color(list(rgb))
                non_target_index += 1

        return [rtn0, rtn1, rtn2]

    def variation_count(self) -> int:
        return min(np.minimum(len(self.sequences), self.max_variations), 200)

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
        index = self.button_indices[self.buttons_pushed]
        x, y, z = self.target_buttons[index].get_position()
        w0.set_position([x, y, z + 0.083])
        w0.set_orientation([math.pi, 0, math.pi])

    def _repeat(self):
        self.buttons_pushed += 1
        return self.buttons_pushed < self.buttons_to_push
