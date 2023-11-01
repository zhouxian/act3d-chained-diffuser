from typing import List, Tuple
import math
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.const import state_size, shape_size


class StackWine(Task):
    def init_task(self):
        wine_bottle = Shape("wine_bottle")
        self.register_graspable_objects([wine_bottle])
        self.register_success_conditions(
            [DetectedCondition(wine_bottle, ProximitySensor("success"))]
        )

    def init_episode(self, index: int) -> List[str]:
        return [
            "stack wine bottle",
            "slide the bottle onto the wine rack",
            "put the wine away",
            "leave the wine on the shelf",
            "grasp the bottle and put it away",
            "place the wine bottle on the wine rack",
        ]

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return ((0.0, 0.0, -math.pi / 4.0), (0.0, 0.0, math.pi / 4.0))

    @property
    def state(self) -> np.ndarray:
        """
        Return a vector containing information for all objects in the scene
        """
        shapes = [Shape("wine_bottle"), ProximitySensor("success")]
        # sort objects according to their x coord
        shapes = sorted(shapes, key=_get_x_coord_from_shape)

        info = np.concatenate([_get_shape_pose(shape) for shape in shapes])

        state = np.zeros(state_size)
        state[: info.size] = info

        return state


def _get_x_coord_from_shape(shape: Shape) -> float:
    return float(shape.get_position()[0])


def _get_shape_pose(shape: Object) -> np.ndarray:
    shape_state = np.concatenate([shape.get_position(), shape.get_quaternion()])
    pad_length = shape_size - shape_state.size
    assert pad_length >= 0
    return np.pad(shape_state, (0, pad_length))
