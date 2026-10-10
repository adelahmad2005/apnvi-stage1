"""
Start Stage 1 and Stage 2 together with one command.

On the Jetson, with the campus PC running Ollama and the speech server:
  ros2 launch apnvi_stage2 stage2.launch.py pc:=<campus PC IP address>
On a laptop with no hardware (typed questions only, Stage 1 in test mode):
  ros2 launch apnvi_stage2 stage2.launch.py camera:=false test_mode:=true pc:=<IP>
Options (Stage 1's options all work here too, see stage1.launch.py):
  pc:=10.1.2.3          the campus PC's IP address (Qwen on 11434, speech on 8090)
  mic:=plughw:2,0       the microphone, as listed by "arecord -l" (default: 'default')
  answer_voice:=false   do not speak Stage 2's sentences (on by default)
  stage2_view:=false    do not start the Stage 2 browser view (port 8081)
Type a question instead of speaking:
  ros2 topic pub --once /voice/question std_msgs/String "{data: 'where is the exit'}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

STAGE1_OPTIONS = ['camera', 'test_mode', 'port', 'rviz', 'view', 'voice', 'camera_height']


def generate_launch_description():
    pc = LaunchConfiguration('pc')
    mic = LaunchConfiguration('mic')
    answer_voice = LaunchConfiguration('answer_voice')
    stage2_view = LaunchConfiguration('stage2_view')
    color_topic = LaunchConfiguration('color_topic')

    # Stage 1 exactly as before, with any Stage 1 option given on the command line.
    stage1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('apnvi_stage1'), 'launch', 'stage1.launch.py'])),
        launch_arguments={name: LaunchConfiguration(name) for name in STAGE1_OPTIONS}.items())

    return LaunchDescription([
        DeclareLaunchArgument('pc', default_value=''),
        DeclareLaunchArgument('mic', default_value='default'),
        DeclareLaunchArgument('answer_voice', default_value='true'),
        DeclareLaunchArgument('stage2_view', default_value='true'),
        DeclareLaunchArgument('color_topic', default_value='/camera/camera/color/image_raw'),
        # Stage 1's options, with Stage 1's own defaults.
        DeclareLaunchArgument('camera', default_value='true'),
        DeclareLaunchArgument('test_mode', default_value='false'),
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('view', default_value='true'),
        DeclareLaunchArgument('voice', default_value='false'),
        DeclareLaunchArgument('camera_height', default_value='1.0'),

        stage1,
        Node(package='apnvi_stage2', executable='stage2_node', output='screen',
             parameters=[{'pc': pc, 'mic': mic, 'voice': answer_voice,
                          'color_topic': color_topic}]),
        Node(package='apnvi_stage2', executable='stage2_view', output='screen',
             parameters=[{'color_topic': color_topic}],
             condition=IfCondition(stage2_view)),
    ])
