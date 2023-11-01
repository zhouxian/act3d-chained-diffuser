"""
This file is not rendering tasks. It is only storing states
"""
from multiprocessing import Process, Manager
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task

import os
import pickle
from rlbench.backend.const import *
import numpy as np

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("save_path", "/tmp/rlbench_data/", "Where to save the demos.")
flags.DEFINE_list(
    "tasks", [], "The tasks to collect. If empty, all tasks are collected."
)
flags.DEFINE_integer(
    "processes", 1, "The number of parallel processes during collection."
)
flags.DEFINE_integer(
    "episodes_per_task", 10, "The number of episodes to collect per task."
)
flags.DEFINE_integer(
    "variations", -1, "Number of variations to collect per task. -1 for all."
)
flags.DEFINE_integer("offset", 0, "First variation id.")


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_demo(demo, example_path):
    for obs in demo:
        # Make sure we don't save images
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    check_and_make(example_path)
    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), "wb") as f:
        pickle.dump(demo, f)


def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    obs_config = ObservationConfig(state=True)
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True,
    )
    rlbench_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ""

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print("Process", i, "finished")
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = FLAGS.offset
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print("Process", i, "finished")
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)
        obs, descriptions = task_env.reset()

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(), VARIATIONS_FOLDER % my_variation_count
        )
        print(variation_path)

        check_and_make(variation_path)

        with open(os.path.join(variation_path, VARIATION_DESCRIPTIONS), "wb") as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task):
            print(
                "Process",
                i,
                "// Task:",
                task_env.get_name(),
                "// Variation:",
                my_variation_count,
                "// Demo:",
                ex_idx,
            )
            attempts = 10
            while attempts > 0:
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                if os.path.exists(episode_path):
                    break
                try:
                    # TODO: for now we do the explicit looping.
                    (demo,) = task_env.get_demos(amount=1, live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        "Process %d failed collecting task %s (variation: %d, "
                        "example: %d). Skipping this task/variation.\n%s\n"
                        % (i, task_env.get_name(), my_variation_count, ex_idx, str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                with file_lock:
                    save_demo(demo, episode_path)
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError("Task %s not recognised!." % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value("i", 0)
    variation_count = manager.Value("i", FLAGS.offset)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    processes = [
        Process(
            target=run,
            args=(i, lock, task_index, variation_count, result_dict, file_lock, tasks),
        )
        for i in range(FLAGS.processes)
    ]
    [t.start() for t in processes]
    [t.join() for t in processes]

    print("Data collection done!")
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == "__main__":
    app.run(main)
