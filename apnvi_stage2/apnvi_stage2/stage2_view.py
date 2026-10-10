"""
Stage 2 live view in the web browser: the colour camera, and the red box after a question.

Test and demo helper, like Stage 1's live view. Shows the live colour picture. After a
question it shows the picture that was asked about, with the red box if found, for
SHOW_ANSWER_S seconds, then goes back to live. Underneath: listening / thinking / ready,
the last question, and what happened.

Open http://localhost:8081, or from a laptop on the same wifi: http://<the Jetson's IP>:8081
Stage 1's live view stays on port 8080, unchanged.

Settings (--ros-args -p name:=value):
  color_topic   default /camera/camera/color/image_raw
  port          default 8081
"""

from http.server import ThreadingHTTPServer
import json
import threading
import time

from apnvi_stage1.hazard_node import NEWEST_PICTURE
from apnvi_stage1.live_view import make_handler
import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .stage2_node import image_to_bgr

PORT = 8081
WIDTH = 960                             # every frame is scaled to this width
SHOW_ANSWER_S = 10.0                    # how long the answer picture stays up
STATUS_COLOURS = {'listening': (0, 200, 230), 'thinking': (0, 130, 255),
                  'ready': (80, 170, 60)}


def fit_width(picture, width=WIDTH):
    """Scale a picture to the given width, keeping its shape."""
    height = round(picture.shape[0] * width / picture.shape[1])
    return cv2.resize(picture, (width, height), interpolation=cv2.INTER_AREA)


def answer_lines(result):
    """Turn the /target/result JSON (as a dict) into two lines of text for the panel."""
    if not result:
        return 'No question yet. Hold the button and ask.', ''
    question = f'Asked: "{result.get("question", "")}"'
    seconds = result.get('seconds')
    took = f' in {seconds:.1f} s' if isinstance(seconds, (int, float)) else ''
    if result.get('found'):
        return question, f'FOUND: {result.get("label") or "it"}{took} (silent)'
    said = f'NOT FOUND{took}. Said: "{result.get("said", "")}"'
    problem = result.get('problem')
    return question, said + (f' ({problem})' if problem else '')


def render(live_bgr, answer_bgr, result, answer_age_s, status):
    """Draw one frame: the picture on top, a text panel underneath."""
    showing_answer = answer_bgr is not None and answer_age_s < SHOW_ANSWER_S
    source = answer_bgr if showing_answer else live_bgr
    if source is None:
        picture = np.zeros((540, WIDTH, 3), np.uint8)
        cv2.putText(picture, 'waiting for the colour camera...', (40, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    else:
        picture = fit_width(np.asarray(source))
    if showing_answer:
        cv2.putText(picture, f'ANSWER (back to live in {SHOW_ANSWER_S - answer_age_s:.0f} s)',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    panel = np.full((130, WIDTH, 3), 30, np.uint8)
    state = status or 'ready'
    cv2.rectangle(panel, (0, 0), (230, 130), STATUS_COLOURS.get(state, (120, 120, 120)), -1)
    cv2.putText(panel, state.upper(), (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 255, 255), 3)
    first, second = answer_lines(result)
    cv2.putText(panel, first[:60], (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    found = bool(result and result.get('found'))
    cv2.putText(panel, second[:70], (250, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (80, 220, 80) if found else (80, 80, 240), 2)
    return np.vstack([picture, panel])


class Stage2View(Node):
    """Collect the newest pictures and answer, and keep a JPEG of the newest frame."""

    def __init__(self):
        super().__init__('stage2_view')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('port', PORT)
        self.port = int(self.get_parameter('port').value)
        self.live_msg = None
        self.answer, self.answer_time = None, 0.0
        self.result, self.status = None, 'ready'
        self.jpeg = None
        self.lock = threading.Lock()
        self.create_subscription(Image, self.get_parameter('color_topic').value,
                                 self.on_live, NEWEST_PICTURE)
        self.create_subscription(Image, '/target/image', self.on_answer, 1)
        self.create_subscription(String, '/target/result', self.on_result, 10)
        self.create_subscription(String, '/stage2/status', self.on_status, 10)
        self.create_timer(1 / 10, self.draw)                 # 10 frames per second

    def on_live(self, msg):
        self.live_msg = msg                                  # convert only when drawing

    def on_answer(self, msg):
        try:
            self.answer, self.answer_time = image_to_bgr(msg), time.monotonic()
        except ValueError as problem:
            self.get_logger().error(str(problem))

    def on_result(self, msg):
        try:
            self.result = json.loads(msg.data)
        except ValueError:
            self.result = None

    def on_status(self, msg):
        self.status = msg.data

    def draw(self):
        live = None
        if self.live_msg is not None:
            try:
                live = image_to_bgr(self.live_msg)
            except ValueError as problem:
                self.get_logger().error(str(problem), throttle_duration_sec=10.0)
        frame = render(live, self.answer, self.result,
                       time.monotonic() - self.answer_time, self.status)
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.jpeg = encoded.tobytes()


def main(args=None):
    """Start the Stage 2 view and its small web server, until Ctrl+C."""
    rclpy.init(args=args)
    view = Stage2View()
    server = ThreadingHTTPServer(('0.0.0.0', view.port), make_handler(view))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    view.get_logger().info(f'Stage 2 view: open http://localhost:{view.port} '
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
