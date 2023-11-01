from typing import List
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import ConditionSet, DetectedCondition, NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.const import state_size, shape_size


class PutKnifeOnChoppingBoard(Task):
    def init_task(self) -> None:
        knife = Shape("knife")
        self._knife_block = Shape("knife_block")
        self._boundary = SpawnBoundary([Shape("boundary")])
        self.register_graspable_objects([knife])
        self.register_success_conditions(
            [
                ConditionSet(
                    [
                        DetectedCondition(knife, ProximitySensor("success")),
                        NothingGrasped(self.robot.gripper),
                    ],
                    order_matters=True,
                )
            ]
        )

    def init_episode(self, index: int) -> List[str]:
        self._boundary.clear()
        self._boundary.sample(self._knife_block)
        return [
            "put the knife on the chopping board",
            "slide the knife out of the knife block and put it down on the "
            "chopping board",
            "place the knife on the chopping board",
            "pick up the knife and leave it on the chopping board",
            "move the knife from the holder to the chopping board",
        ]

    def variation_count(self) -> int:
        return 1

    @property
    def state(self) -> np.ndarray:
        """
        Return a vector containing information for all objects in the scene
        """
        if not hasattr(self, "_knife_block"):
            raise RuntimeError("Please initialize the task first")

        shapes = [self._knife_block, Shape("knife")]
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
