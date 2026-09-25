"""
hazard_node: the program that decides how dangerous things are.

Listens to:  the camera depth channel (setting: depth_topic) and /ultrasonic/distance
Sends, about 10 times a second:
  /hazard/distance  Float32  metres to the nearest thing ahead, -1 = nothing in range
  /hazard/level     String   clear / notice / warning / stop
  /sensor/status    String   camera_ok / camera_blocked

All the maths lives in hazard_check.py (plain Python, tested without ROS).
"""

import time

from apnvi_stage1.hazard_check import check_depth, level_with_dead_zone, nearest_distance
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

DEPTH_ENCODINGS = ('16UC1', 'mono16')   # one 16-bit whole number per pixel, in mm


def image_to_depth_mm(msg):
    """Turn a ROS depth Image message into a 2D numpy grid of millimetres."""
    if msg.encoding not in DEPTH_ENCODINGS:
        raise ValueError(f'expected a 16-bit depth picture, got encoding "{msg.encoding}"')
    byte_order = '>' if msg.is_bigendian else '<'
    pixels_per_row = msg.step // 2                  # step = bytes per row, 2 bytes per pixel
    grid = np.frombuffer(msg.data, dtype=byte_order + 'u2')
    grid = grid.reshape(msg.height, pixels_per_row)
    return grid[:, :msg.width]                      # drop any padding at the end of each row


class HazardNode(Node):
    """Combine the camera and the ultrasonic into one distance, one level and one status."""

    def __init__(self):
        super().__init__('hazard_node')

        # Settings. Change them at start-up with: --ros-args -p depth_topic:=/other/name
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('stale_after_s', 0.5)      # older readings count as missing
        self.declare_parameter('startup_wait_s', 5.0)     # time for the camera to start
        depth_topic = self.get_parameter('depth_topic').value
        rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.stale_after_s = float(self.get_parameter('stale_after_s').value)
        self.startup_wait_s = float(self.get_parameter('startup_wait_s').value)

        # Latest reading from each sensor, and when it arrived (None = never).
        self.camera_m = None
        self.camera_blocked = True
        self.camera_time = None
        self.ultrasonic_m = None
        self.ultrasonic_time = None

        self.level = None               # previous level, for the dead zone
        self.last_report = None         # previous (level, status), to log changes only
        self.start_time = time.monotonic()

        # "Newest is fine" receiving mode: works whether the sender is reliable or best-effort.
        self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(
            Float32, '/ultrasonic/distance', self.on_ultrasonic, qos_profile_sensor_data)

        self.distance_pub = self.create_publisher(Float32, '/hazard/distance', 10)
        self.level_pub = self.create_publisher(String, '/hazard/level', 10)
        self.status_pub = self.create_publisher(String, '/sensor/status', 10)

        self.create_timer(1.0 / rate_hz, self.on_timer)
        self.get_logger().info(
            f'hazard_node started: camera on "{depth_topic}", ultrasonic on '
            f'"/ultrasonic/distance", sending {rate_hz:.0f} times per second')

    def on_depth(self, msg):
        """Run the check on each new camera picture and remember the result."""
        try:
            depth_mm = image_to_depth_mm(msg)
        except ValueError as error:
            self.get_logger().warn(str(error), throttle_duration_sec=5.0)
            return
        distance_m, blocked, _ = check_depth(depth_mm)
        self.camera_m = distance_m
        self.camera_blocked = blocked
        self.camera_time = time.monotonic()

    def on_ultrasonic(self, msg):
        """Remember the latest ultrasonic distance (-1 = nothing in range)."""
        self.ultrasonic_m = float(msg.data)
        self.ultrasonic_time = time.monotonic()

    def is_fresh(self, arrived_at):
        """Return True if a reading arrived recently enough to trust."""
        return arrived_at is not None and time.monotonic() - arrived_at <= self.stale_after_s

    def on_timer(self):
        """Decide and send distance, level and status (runs about 10 times a second)."""
        # Stay quiet while the camera driver starts up, so we do not say "blocked" at start.
        still_starting = time.monotonic() - self.start_time < self.startup_wait_s
        if self.camera_time is None and still_starting:
            return

        camera_ok = self.is_fresh(self.camera_time) and not self.camera_blocked
        camera_m = self.camera_m if camera_ok else None
        ultrasonic_m = self.ultrasonic_m if self.is_fresh(self.ultrasonic_time) else None

        distance_m = nearest_distance(camera_m, ultrasonic_m)
        if distance_m < 0:
            self.level = 'clear'                    # nothing in range from either sensor
        else:
            self.level = level_with_dead_zone(distance_m, self.level)
        status = 'camera_ok' if camera_ok else 'camera_blocked'

        self.distance_pub.publish(Float32(data=float(distance_m)))
        self.level_pub.publish(String(data=self.level))
        self.status_pub.publish(String(data=status))

        if (self.level, status) != self.last_report:
            self.get_logger().info(f'{self.level}, {status}, distance {distance_m:.2f} m')
            self.last_report = (self.level, status)


def main(args=None):
    """Start the node and keep it running until Ctrl+C."""
    rclpy.init(args=args)
    node = HazardNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass                                        # normal stop (Ctrl+C), not an error
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
