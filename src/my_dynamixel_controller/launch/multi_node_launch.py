from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_dynamixel_controller',
            executable='arduino_controller_node',
            name='arduino_controller',
            output='screen'
        ),
        Node(
            package='my_dynamixel_controller',
            executable='call_logger_client',
            name='logger_client',
            output='screen'
        )
    ])
