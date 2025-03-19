import numpy as np

from composuite.arenas.pick_place_arena import PickPlaceArena
from composuite.env.compositional_env import CompositionalEnv

import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.models.tasks import ManipulationTask
from .pick_place_subtask import PickPlaceSubtask


class ApproachSubtask(PickPlaceSubtask):
    """Hover Task: The agent must move the object above the target bin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subtask_id = 7

    def staged_rewards(self, action):
        """
        Compute rewards based on hovering behavior.
        """
        hover_mult = 1.0
        r_hover = 0.0

        # Compute distance to goal bin
        object_xy_loc = self.sim.data.body_xpos[self.obj_body_id, :2]
        dist = np.linalg.norm(self.bin2_pos[:2] - object_xy_loc)

        # Reward based on closeness to the goal
        r_hover = (1 - np.tanh(2.0 * dist)) * hover_mult

        return 0, 0, 0, r_hover, 0  # Only hovering reward

    def _check_success(self):
        """Task is successful when object is above the target bin."""
        object_xy_loc = self.sim.data.body_xpos[self.obj_body_id, :2]
        return np.linalg.norm(self.bin2_pos[:2] - object_xy_loc) < 0.05