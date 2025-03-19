import numpy as np

from composuite.arenas.pick_place_arena import PickPlaceArena
from composuite.env.compositional_env import CompositionalEnv

import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.models.tasks import ManipulationTask
from .pick_place_subtask import PickPlaceSubtask


class ReachSubtask(PickPlaceSubtask):
    """Reach Task: The agent must move its gripper to the target object."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subtask_id = 4

    def staged_rewards(self, action):
        """
        Compute rewards based on reaching behavior.
        """
        reach_mult = 1.0  # Full weight to reaching
        r_reach = 0.0

        # Compute distance between gripper and target object
        dist = self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.object.root_body,
            target_type="body",
            return_distance=True,
        )

        # Reward is inverse of distance (closer = higher reward)
        r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult

        return r_reach, 0, 0, 0, 0  # Only returning reach reward

    def _check_success(self):
        """Task is successful when gripper is close to object."""
        return self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.object.root_body,
            target_type="body",
            return_distance=True,
        ) < 0.05  # Success if within 5cm