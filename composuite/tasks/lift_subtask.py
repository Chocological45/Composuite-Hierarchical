import numpy as np

from composuite.arenas.pick_place_arena import PickPlaceArena
from composuite.env.compositional_env import CompositionalEnv

import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.models.tasks import ManipulationTask
from .pick_place_subtask import PickPlaceSubtask


class LiftSubtask(PickPlaceSubtask):
    """Lift Task: The agent must lift the object upwards."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subtask_id = 6

    def staged_rewards(self, action):
        """
        Compute rewards based on lifting behavior.
        """
        lift_mult = 1.0
        r_lift = 0.0

        # Check if the object is grasped
        is_grasping = self._check_grasp(self.robots[0].gripper, self.object.contact_geoms)
        if not is_grasping:
            return 0, 0, 0, 0, 0  # No reward if not grasping

        # Compute lifting reward
        object_z_loc = self.sim.data.body_xpos[self.obj_body_id, 2]
        target_z = self.bin2_pos[2] + 0.25
        z_dist = np.abs(target_z - object_z_loc)
        r_lift = (1 - np.tanh(5.0 * z_dist)) * lift_mult

        return 0, 0, r_lift, 0, 0  # Only lifting reward

    def _check_success(self):
        """Task is successful when object is lifted above a threshold."""
        object_z_loc = self.sim.data.body_xpos[self.obj_body_id, 2]
        return object_z_loc > self.bin2_pos[2] + 0.15