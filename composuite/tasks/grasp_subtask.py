import numpy as np

from composuite.arenas.pick_place_arena import PickPlaceArena
from composuite.env.compositional_env import CompositionalEnv

import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.models.tasks import ManipulationTask
from .pick_place_subtask import PickPlaceSubtask


class GraspSubtask(PickPlaceSubtask):
    """Grasp Task: The agent must grasp the object."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subtask_id = 5

    def staged_rewards(self, action):
        """
        Compute rewards based on grasping behavior.
        """
        grasp_mult = 1.0  # Full weight to grasping
        r_grasp = 0.0

        # Check if gripper is grasping the object
        is_grasping = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.object.contact_geoms])

        r_grasp = int(is_grasping) * grasp_mult

        return 0, r_grasp, 0, 0, 0  # Only grasping reward

    def _check_success(self):
        """Task is successful when object is grasped."""
        return self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.object.contact_geoms])