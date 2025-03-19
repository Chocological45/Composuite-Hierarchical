import numpy as np

from composuite.arenas.pick_place_arena import PickPlaceArena
from composuite.env.compositional_env import CompositionalEnv

import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.models.tasks import ManipulationTask
from .pick_place_subtask import PickPlaceSubtask


class LowerSubtask(PickPlaceSubtask):
    """Place Task: The agent must drop the object into the bin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subtask_id = 8

    def staged_rewards(self, action):
        """
        Compute rewards based on placing behavior.
        """
        drop_mult = 1.0
        r_drop = 0.0

        # Check if the object is above the bin
        object_xy_loc = self.sim.data.body_xpos[self.obj_body_id, :2]
        is_above_bin = np.linalg.norm(self.bin2_pos[:2] - object_xy_loc) < 0.05

        # Reward based on object dropping
        if is_above_bin:
            r_drop = drop_mult

        return 0, 0, 0, 0, r_drop  # Only placing reward

    def _check_success(self):
        """Task is successful when object is inside the bin."""
        return not self.not_in_bin(self.sim.data.body_xpos[self.obj_body_id])