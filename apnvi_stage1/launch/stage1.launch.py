"""
Start the whole Stage 1 system with one command.

On the Jetson (real camera + real ESP32):
  ros2 launch apnvi_stage1 stage1.launch.py
On a laptop with no hardware (play a clip in another window with tools/play_clip.py):
  ros2 launch apnvi_stage1 stage1.launch.py camera:=false test_mode:=true
Options:
  camera:=false        do not start the RealSense driver
  test_mode:=true      made-up ultrasonic distances, no ESP32 needed
  port:=/dev/ttyACM0   the ESP32's USB port (default /dev/ttyUSB0)
  rviz:=true           also open rviz2 showing the depth picture
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera = LaunchConfiguration('camera')
    test_mode = LaunchConfiguration('test_mode')
    port = LaunchConfiguration('port')
    depth_topic = LaunchConfiguration('depth_topic')
    rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value='true'),
        DeclareLaunchArgument('test_mode', default_value='false'),
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('depth_topic',
                              default_value='/camera/camera/depth/image_rect_raw'),
        DeclareLaunchArgument('rviz', default_value='false'),

        # The RealSense camera driver (only on the device, where it is installed).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
            condition=IfCondition(camera)),

        Node(package='apnvi_stage1', executable='esp32_reader', output='screen',
             parameters=[{'port': port, 'test_mode': test_mode}]),
        Node(package='apnvi_stage1', executable='hazard_node', output='screen',
             parameters=[{'depth_topic': depth_topic}]),
        Node(package='apnvi_stage1', executable='output_node', output='screen'),

        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', PathJoinSubstitution(
                 [FindPackageShare('apnvi_stage1'), 'rviz', 'stage1.rviz'])],
             condition=IfCondition(rviz)),
    ])
