import os 
import robosuite
from composuite.tasks.pick_place_subtask import PickPlaceSubtask
from composuite.tasks.push_subtask import PushSubtask
from composuite.tasks.shelf_subtask import ShelfSubtask
from composuite.tasks.trashcan_subtask import TrashcanSubtask
from composuite.tasks.reach_subtask import ReachSubtask
from composuite.tasks.grasp_subtask import GraspSubtask
from composuite.tasks.lift_subtask import LiftSubtask
from composuite.tasks.approach_subtask import ApproachSubtask
from composuite.tasks.lower_subtask import LowerSubtask

robosuite.environments.base.register_env(PickPlaceSubtask)
robosuite.environments.base.register_env(PushSubtask)
robosuite.environments.base.register_env(ShelfSubtask)
robosuite.environments.base.register_env(TrashcanSubtask)

# Pick and place sub tasks
robosuite.environments.base.register_env(ReachSubtask)
robosuite.environments.base.register_env(GraspSubtask)
robosuite.environments.base.register_env(LiftSubtask)
robosuite.environments.base.register_env(ApproachSubtask)
robosuite.environments.base.register_env(LowerSubtask)


from composuite.env.main import make, sample_tasks
assets_root = os.path.join(os.path.dirname(__file__), "assets")
