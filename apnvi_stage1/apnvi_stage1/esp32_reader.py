"""
esp32_reader: reads the ultrasonic distance from the ESP32 over USB.

The ESP32 sends one number per line, in metres with 2 decimals (e.g. 1.23), or -1 when
nothing is in range, about 10 times a second at 115200 baud.

Sends:  /ultrasonic/distance  Float32  metres, -1 = nothing in range

Settings (--ros-args -p name:=value):
  port       USB port of the ESP32, default /dev/ttyUSB0
  test_mode  true = no ESP32 needed: makes up distances going 3 m -> 0.3 m -> 3 m, forever
"""

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32

BAUD = 115200
TEST_FAR_M, TEST_NEAR_M = 3.0, 0.3      # test mode range
TEST_SPEED_M_PER_S = 0.3                # test mode: 9 s from far to near


def parse_line(line):
    """Turn one line from the ESP32 into metres, or None if it is not a number."""
    try:
        value = float(line.strip())
    except ValueError:
        return None
    return -1.0 if value < 0 else value


def fake_distance(seconds):
    """Made-up distance for test mode: goes slowly from 3 m to 0.3 m and back, forever."""
    span = TEST_FAR_M - TEST_NEAR_M
    travelled = (seconds * TEST_SPEED_M_PER_S) % (2 * span)
    offset = travelled if travelled <= span else 2 * span - travelled
    return round(TEST_FAR_M - offset, 2)


class Esp32Reader(Node):
    """Send the ESP32's ultrasonic distance (or made-up test distances) on a ROS channel."""

    def __init__(self):
        super().__init__('esp32_reader')
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('test_mode', False)
        self.port = self.get_parameter('port').value
        self.test_mode = bool(self.get_parameter('test_mode').value)

        self.pub = self.create_publisher(Float32, '/ultrasonic/distance', 10)
        self.serial = None
        self.buffer = b''
        self.start = time.monotonic()
        self.last_open_try = 0.0

        if self.test_mode:
            self.create_timer(0.1, self.send_fake_distance)
            self.get_logger().info('TEST MODE: made-up distances 3 m -> 0.3 m -> 3 m')
        else:
            self.create_timer(0.02, self.read_serial)
            self.get_logger().info(f'Reading the ESP32 on {self.port} at {BAUD} baud')

    def send_fake_distance(self):
        """Test mode: send the next made-up distance."""
        self.pub.publish(Float32(data=fake_distance(time.monotonic() - self.start)))

    def open_port(self):
        """Try to open the USB port. If it fails (ESP32 not plugged in), try again in 2 s."""
        if time.monotonic() - self.last_open_try < 2.0:
            return
        self.last_open_try = time.monotonic()
        try:
            import serial                       # python3-serial, only needed with a real ESP32
            self.serial = serial.Serial(self.port, BAUD, timeout=0)
            self.get_logger().info(f'Connected to the ESP32 on {self.port}')
        except Exception as error:              # any failure: report it and keep trying
            self.serial = None
            self.get_logger().warn(f'Cannot open {self.port} ({error}). Retrying...',
                                   throttle_duration_sec=10.0)

    def read_serial(self):
        """Read whatever the ESP32 has sent, and send each complete number on."""
        if self.serial is None:
            self.open_port()
            return
        try:
            self.buffer += self.serial.read(self.serial.in_waiting or 1)
        except Exception as error:              # unplugged while running
            self.get_logger().warn(f'Lost the ESP32 ({error}). Reconnecting...')
            self.serial = None
            return
        *lines, self.buffer = self.buffer.split(b'\n')   # keep any half line for next time
        for line in lines:
            value = parse_line(line.decode('ascii', errors='ignore'))
            if value is not None:               # skip anything that is not a number
                self.pub.publish(Float32(data=value))


def main(args=None):
    """Start the node and keep it running until Ctrl+C."""
    rclpy.init(args=args)
    node = Esp32Reader()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
