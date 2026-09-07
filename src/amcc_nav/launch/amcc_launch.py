from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('amcc_nav'),
        'config', 'amcc_params.yaml'
    )

    amcc_detector = Node(
        package='amcc_nav',
        executable='amcc_detector',
        name='amcc_detector',
        parameters=[config],
        output='screen'
    )

    hybrid_map_builder = Node(
        package='amcc_nav',
        executable='hybrid_map_builder',
        name='hybrid_map_builder',
        parameters=[config],
        output='screen'
    )

    return LaunchDescription([
        amcc_detector,
        hybrid_map_builder,
    ])