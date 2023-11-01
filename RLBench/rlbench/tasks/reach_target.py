from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors, state_size, shape_size
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition


class ReachTarget(Task):
    def init_task(self) -> None:
        self.target = Shape("target")
        self.distractor0 = Shape("distractor0")
        self.distractor1 = Shape("distractor1")
        self.boundaries = Shape("boundary")
        success_sensor = ProximitySensor("success")
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)]
        )

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2,
            replace=False,
        )
        for ob, i in zip([self.distractor0, self.distractor1], color_choices):
            name, rgb = colors[i]
            ob.set_color(rgb)
        b = SpawnBoundary([self.boundaries])
        for ob in [self.target, self.distractor0, self.distractor1]:
            b.sample(
                ob, min_distance=0.2, min_rotation=(0, 0, 0), max_rotation=(0, 0, 0)
            )

        return [
            "reach the %s target" % color_name,
            "touch the %s ball with the panda gripper" % color_name,
            "reach the %s sphere" % color_name,
        ]

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.array(self.target.get_position())

    def is_static_workspace(self) -> bool:
        return True

    @property
    def state(self) -> np.ndarray:
        """
        Return a vector containing information for all objects in the scene
        """
        if not hasattr(self, "target"):
            raise RuntimeError("Please initialize the task first")

        shapes = [self.target, self.distractor1, self.distractor0]
        # sort objects according to their x coord
        shapes = sorted(shapes, key=_get_color)

        info = np.concatenate([_get_shape_info(shape) for shape in shapes])

        state = np.zeros(state_size)
        state[: info.size] = info

        return state


def _get_color(shape: Shape) -> List[float]:
    return list(shape.get_color())


def _get_shape_info(shape: Shape) -> np.ndarray:
    color = np.asarray(shape.get_color())
    shape_state = np.concatenate([shape.get_position(), shape.get_quaternion(), color])
    pad_length = shape_size - shape_state.size
    assert pad_length >= 0
    return np.pad(shape_state, (0, pad_length))
