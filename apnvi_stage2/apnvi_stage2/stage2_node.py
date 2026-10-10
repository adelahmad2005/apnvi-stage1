"""
stage2_node: the user holds the button, asks for something, and gets a red box or a sentence.

While the button is held, the microphone records. When it is let go, the node takes the
latest colour picture, sends the recording and the picture to the campus PC, and:
  found      -> publishes the picture with the red box drawn on it, stays silent
  not found  -> says "I can't see it, turn slowly and try again"
The work runs in the background, so Stage 1's warnings never wait for it.
One question at a time: a press while a question is being answered is ignored.

Listens: /button/pressed   Bool    from esp32_reader (True = held, False = let go)
         /voice/question   String  a typed question, for testing without a microphone:
             ros2 topic pub --once /voice/question std_msgs/String "{data: 'where is the exit'}"
         colour picture    Image   from the RealSense (setting color_topic)
Sends:   /target/result    String  JSON: question, found, box [left, top, right, bottom],
                                    label, seconds, said, problem
         /target/image     Image   the picture the question was asked about, with the box

Settings (--ros-args -p name:=value):
  ollama_url    where Qwen runs, default http://localhost:11434
  speech_url    where the speech server runs, default http://localhost:8090
  model         default qwen3-vl:4b
  color_topic   default /camera/camera/color/image_raw
  mic           microphone for arecord, default 'default'
  voice         true = speak the sentences (espeak-ng), default true
"""

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from .asking import answer_recording, answer_typed
from .finder import MODEL, OLLAMA_URL
from .speech import SPEECH_URL

MIN_HOLD_S = 0.3            # shorter than this = an accidental tap, ignored
MAX_HOLD_S = 15.0           # stop recording after this, even if still held
VOICE_SPEED = '160'         # espeak-ng words per minute, same as Stage 1


def image_to_bgr(msg):
    """Turn a ROS colour picture (rgb8 or bgr8) into a numpy grid in OpenCV's order."""
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f'expected an rgb8 or bgr8 picture, got "{msg.encoding}"')
    grid = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    grid = grid[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return grid[:, :, ::-1].copy() if msg.encoding == 'rgb8' else grid.copy()


def bgr_to_image(picture, stamp=None):
    """Turn a numpy picture in OpenCV's order into a ROS bgr8 picture."""
    msg = Image()
    msg.height, msg.width = picture.shape[:2]
    msg.encoding = 'bgr8'
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(picture, dtype=np.uint8).tobytes()
    if stamp is not None:
        msg.header.stamp = stamp
    return msg


def result_json(outcome):
    """Pack an Outcome into the JSON sent on /target/result."""
    result = outcome.result
    return json.dumps({
        'question': outcome.question,
        'found': bool(result and result.found),
        'box': list(result.box) if result and result.found else None,
        'label': result.label if result else '',
        'seconds': round(result.seconds, 2) if result else None,
        'said': outcome.say,
        'problem': outcome.problem,
    })


class Recorder:
    """Records the microphone with arecord into a WAV file, between start() and stop()."""

    def __init__(self, mic='default'):
        self.mic = mic
        self.process = None
        self.path = None
        self.started = 0.0

    def start(self):
        """Start recording: 16 kHz, mono, 16-bit, which is what Whisper wants."""
        handle, self.path = tempfile.mkstemp(prefix='apnvi_question_', suffix='.wav')
        os.close(handle)
        self.process = subprocess.Popen(
            ['arecord', '-q', '-D', self.mic, '-f', 'S16_LE', '-r', '16000', '-c', '1',
             '-t', 'wav', self.path], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.started = time.monotonic()

    def stop(self):
        """Stop recording. Returns (WAV bytes, seconds held), or (None, seconds) on failure."""
        held = time.monotonic() - self.started
        if self.process is None:
            return None, held
        self.process.send_signal(signal.SIGINT)        # arecord finishes the file cleanly
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None
        try:
            with open(self.path, 'rb') as file:
                data = file.read()
        except OSError:
            data = None
        finally:
            if os.path.exists(self.path):
                os.remove(self.path)
        return (data if data and len(data) > 44 else None), held   # 44 = empty WAV header


class Stage2Node(Node):
    """Hold the button, ask, get a red box or a sentence."""

    def __init__(self):
        super().__init__('stage2_node')
        self.declare_parameter('ollama_url', OLLAMA_URL)
        self.declare_parameter('speech_url', SPEECH_URL)
        self.declare_parameter('model', MODEL)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('mic', 'default')
        self.declare_parameter('voice', True)
        self.ollama_url = self.get_parameter('ollama_url').value
        self.speech_url = self.get_parameter('speech_url').value
        self.model = self.get_parameter('model').value
        self.voice = bool(self.get_parameter('voice').value)
        self.espeak = shutil.which('espeak-ng')
        if self.voice and not self.espeak:
            self.get_logger().warn('espeak-ng not found: sentences will only be printed')

        self.recorder = Recorder(self.get_parameter('mic').value)
        if not shutil.which('arecord'):
            self.get_logger().warn('arecord not found: the button cannot record. '
                                   'Typed questions on /voice/question still work')
        self.latest = None                      # latest colour picture message
        self.busy = threading.Lock()            # one question at a time
        self.recording = False

        self.result_pub = self.create_publisher(String, '/target/result', 10)
        self.image_pub = self.create_publisher(Image, '/target/image', 1)
        self.create_subscription(Image, self.get_parameter('color_topic').value,
                                 self.on_picture, qos_profile_sensor_data)
        self.create_subscription(Bool, '/button/pressed', self.on_button, 10)
        self.create_subscription(String, '/voice/question', self.on_typed, 10)
        self.create_timer(0.5, self.check_long_hold)
        self.get_logger().info(f'Ready. Qwen at {self.ollama_url}, speech at {self.speech_url}')

    def on_picture(self, msg):
        """Keep only the newest colour picture."""
        self.latest = msg

    def latest_bgr(self):
        """Return the newest colour picture as numpy, or None if there is none yet."""
        if self.latest is None:
            return None
        try:
            return image_to_bgr(self.latest)
        except ValueError as problem:
            self.get_logger().error(str(problem), throttle_duration_sec=10.0)
            return None

    def on_button(self, msg):
        """Button held: start recording. Let go: stop and answer."""
        if msg.data and not self.recording:
            if self.busy.locked():
                self.get_logger().info('Still answering the last question: press ignored')
                return
            try:
                self.recorder.start()
            except OSError as problem:
                self.get_logger().error(f'Cannot start the microphone: {problem}')
                return
            self.recording = True
            self.get_logger().info('Listening...')
        elif not msg.data and self.recording:
            self.finish_recording()

    def check_long_hold(self):
        """Stop recording if the button has been held for too long."""
        if self.recording and time.monotonic() - self.recorder.started > MAX_HOLD_S:
            self.get_logger().info(f'Held for over {MAX_HOLD_S:.0f} s: answering now')
            self.finish_recording()

    def finish_recording(self):
        """Stop the microphone and answer in the background."""
        self.recording = False
        wav, held = self.recorder.stop()
        image = self.latest_bgr()               # the picture at the moment of letting go
        if held < MIN_HOLD_S:
            self.get_logger().info('Button tapped too briefly: ignored')
            return
        if wav is None:
            self.get_logger().error('Nothing was recorded: is the microphone plugged in?')
            return
        self.start_answer(lambda: answer_recording(
            wav, image, self.speech_url, self.ollama_url, self.model))

    def on_typed(self, msg):
        """Answer a typed question (testing without a microphone)."""
        if self.busy.locked():
            self.get_logger().info('Still answering the last question: typed one ignored')
            return
        image = self.latest_bgr()
        self.start_answer(lambda: answer_typed(msg.data, image, self.ollama_url, self.model))

    def start_answer(self, work):
        """Run the answer in a background thread so nothing else waits for it."""
        self.busy.acquire()

        def run():
            try:
                outcome = work()
                self.deliver(outcome)
            except Exception as problem:        # never let one question kill the node
                self.get_logger().error(f'Answering failed: {problem}')
            finally:
                self.busy.release()

        threading.Thread(target=run, daemon=True).start()

    def deliver(self, outcome):
        """Publish the result and the picture, and speak if there is something to say."""
        self.result_pub.publish(String(data=result_json(outcome)))
        if outcome.picture is not None:
            self.image_pub.publish(bgr_to_image(outcome.picture))
        found = outcome.result is not None and outcome.result.found
        self.get_logger().info(
            f'"{outcome.question}": ' + (f'found {outcome.result.label}' if found else
                                         f'said "{outcome.say}"') +
            (f' ({outcome.problem})' if outcome.problem else ''))
        if outcome.say and self.voice and self.espeak:
            subprocess.Popen([self.espeak, '-s', VOICE_SPEED, outcome.say],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(args=None):
    """Start the node and keep it running until Ctrl+C."""
    rclpy.init(args=args)
    node = Stage2Node()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
