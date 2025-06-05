import os

from teleop_demo_python.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from teleop_demo_python.utils.path_utils import ASSET_PATH


def main():
    xml_path = os.path.join(ASSET_PATH, "universal_robots_ur5e/scene_dual_arm.xml")
    robot_urdf_path = os.path.join(ASSET_PATH, "universal_robots_ur5e/dual_ur5e.urdf")

    config = {
        "right_hand": {
            "link_name": "right_tool0",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",
        },
        "left_hand": {
            "link_name": "left_tool0",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "vis_target": "left_target",
        },
    }

    # Create and initialize the teleoperation controller
    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        end_effector_config=config,
        scale_factor=1.5,
        visualize_placo=True,
    )

    controller.run()


if __name__ == "__main__":
    main()
