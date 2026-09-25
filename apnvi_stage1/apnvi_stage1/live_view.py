"""
live_view: shows what the camera sees and what the system decides, in a web browser.

The depth picture in colour (red = near, blue = far, black = no reading), the walking
lane as a white box, and underneath: the level, the distance, the camera status and how
fast it is beeping. Started by stage1.launch.py (view:=true, the default).

Open it in any browser:
  on the same computer:            http://localhost:8080
  from a laptop on the same wifi:  http://<the Jetson's IP address>:8080
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

from apnvi_stage1.output_node import beep_gap
import cv2
import numpy as np
from rcl_interfaces.msg import Log
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

FAR_MM = 4000                                           # colour scale: 0 to 4 m
LEVEL_COLOURS = {'clear': (80, 170, 60), 'notice': (0, 200, 230),      # blue, green, red
                 'warning': (0, 130, 255), 'stop': (40, 40, 220)}
PAGE = (b'<html><head><title>APNVI live view</title></head>'
        b'<body style="background:#222;margin:0;text-align:center">'
        b'<img src="/stream" style="max-width:100%"></body></html>')


def render(depth_mm, level, distance_m, status, said, said_age_s):
    """Draw one frame: coloured depth picture with the lane box, and a text panel below."""
    if depth_mm is None:
        picture = np.zeros((480, 848, 3), np.uint8)
        cv2.putText(picture, 'waiting for the camera...', (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    else:
        near_is_bright = 255 - np.clip(depth_mm.astype(np.int32), 0, FAR_MM) * 255 // FAR_MM
        picture = cv2.applyColorMap(near_is_bright.astype(np.uint8), cv2.COLORMAP_JET)
        picture[depth_mm == 0] = 0                      # no reading -> black
        height, width = depth_mm.shape
        cv2.rectangle(picture, (width // 3, height // 4), (2 * width // 3, 3 * height // 4),
                      (255, 255, 255), 2)
    panel = np.full((150, picture.shape[1], 3), 30, np.uint8)
    colour = LEVEL_COLOURS.get(level, (120, 120, 120))
    cv2.rectangle(panel, (0, 0), (260, 150), colour, -1)
    cv2.putText(panel, (level or '-').upper(), (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (255, 255, 255), 4)
    distance_text = 'nothing in range' if distance_m is None or distance_m < 0 \
        else f'{distance_m:.2f} m'
    cv2.putText(panel, distance_text, (285, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2)
    status_colour = (40, 40, 220) if status == 'camera_blocked' else (200, 200, 200)
    cv2.putText(panel, status or 'waiting', (285, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                status_colour, 2)
    gap = beep_gap(level, -1.0 if distance_m is None else distance_m)
    if gap is None:
        beeps = 'beeps: silent'
    elif level == 'stop':
        beeps = 'beeps: continuous'
    else:
        beeps = f'beeps: every {gap:.1f} s'
    cv2.putText(panel, beeps, (285, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    if said and said_age_s < 3:                     # only if the voice is switched on
        cv2.putText(panel, f'said: "{said}"', (540, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)
    return np.vstack([picture, panel])


class LiveView(Node):
    """Collect the latest picture and decisions, and keep a JPEG of the newest frame."""

    def __init__(self):
        super().__init__('live_view')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('port', 8080)
        depth_topic = self.get_parameter('depth_topic').value
        self.port = int(self.get_parameter('port').value)
        self.depth = None
        self.level, self.distance, self.status = None, None, None
        self.said, self.said_time = None, 0.0
        self.jpeg = None
        self.lock = threading.Lock()
        self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(String, '/hazard/level', self.on_level, 10)
        self.create_subscription(Float32, '/hazard/distance', self.on_distance, 10)
        self.create_subscription(String, '/sensor/status', self.on_status, 10)
        self.create_subscription(Log, '/rosout', self.on_log, 50)   # to catch "SAY: ..."
        self.create_timer(1 / 15, self.draw)                        # 15 frames per second

    def on_depth(self, msg):
        grid = np.frombuffer(msg.data, dtype='<u2').reshape(msg.height, msg.step // 2)
        self.depth = grid[:, :msg.width]

    def on_level(self, msg):
        self.level = msg.data

    def on_distance(self, msg):
        self.distance = msg.data

    def on_status(self, msg):
        self.status = msg.data

    def on_log(self, msg):
        if msg.name == 'output_node' and msg.msg.startswith('SAY: '):
            self.said, self.said_time = msg.msg[5:], time.monotonic()

    def draw(self):
        frame = render(self.depth, self.level, self.distance, self.status,
                       self.said, time.monotonic() - self.said_time)
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.jpeg = encoded.tobytes()


def make_handler(view):
    """Web page handler: '/' is the page, '/stream' is the moving picture."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(PAGE)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with view.lock:
                        jpeg = view.jpeg
                    if jpeg:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                                         + jpeg + b'\r\n')
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass                                # browser tab closed

        def log_message(self, *args):
            pass                                    # keep the terminal quiet

    return Handler


def main(args=None):
    """Start the live view and its small web server, until Ctrl+C."""
    rclpy.init(args=args)
    view = LiveView()
    server = ThreadingHTTPServer(('0.0.0.0', view.port), make_handler(view))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    view.get_logger().info(f'Live view: open http://localhost:{view.port} '
                           f'(or http://<this computer\'s IP>:{view.port}) in a browser')
    try:
        rclpy.spin(view)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        server.shutdown()
        view.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
