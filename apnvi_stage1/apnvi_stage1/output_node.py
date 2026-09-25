"""
output_node: the program that beeps (and can talk, but the voice is OFF by default).

Decided 2026-09-25 (Adel): no voice to the user. Only beeps. The voice can be switched
back on with the setting voice:=true, for example for testing.

Listens to /hazard/level, /hazard/distance and /sensor/status (sent by hazard_node) and:
  - speaks with espeak-ng: "Obstacle ahead, 1.2 metres" at warning, "Stop" at stop,
    "Camera blocked" and "Camera back". Only when something changes, and the same kind of
    message is not repeated within 4 seconds.
  - beeps: silent at clear; otherwise the gap between beeps = distance x 0.5 s;
    almost continuous at stop. Beeps pause while it is speaking.
  - shows a bar from 0 to 100 in the terminal, fuller when closer.

Speech is turned into sound files once, at start-up, and then just played. That makes it
start instantly and play smoothly, instead of being made on the fly while it plays.
"""

import math
import os
import shutil
import subprocess
import tempfile
import time
import wave

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32, String

REPEAT_S = 4.0              # same kind of message at most once every 4 s
BEEP_GAP_PER_M = 0.5        # gap between beeps = distance x 0.5 s
STOP_GAP_S = 0.05           # at stop: almost continuous
BEEP_S = 0.08               # length of one beep
BAR_FAR_M, BAR_NEAR_M = 3.0, 0.3   # bar is empty at 3 m or more, full at 0.3 m or less
INPUT_TIMEOUT_S = 1.0       # nothing from hazard_node for 1 s -> go silent
VOICE_SPEED = '160'         # espeak-ng words per minute
SOUND_DIR = os.path.join(tempfile.gettempdir(), 'apnvi_sounds')


def message_for(level, status, distance_m, old_level, old_status):
    """
    Decide what to say after a change. Returns a list of (kind, text).

    kind is used for the "not again within 4 s" rule, text is what gets spoken.
    """
    messages = []
    if status != old_status:
        if status == 'camera_blocked':
            messages.append(('blocked', 'Camera blocked'))
        elif old_status == 'camera_blocked':
            messages.append(('back', 'Camera back'))
    if level != old_level:
        if level == 'stop':
            messages.append(('stop', 'Stop'))
        elif level == 'warning' and distance_m >= 0:
            messages.append(('warning', f'Obstacle ahead, {distance_m:.1f} metres'))
    return messages


def beep_gap(level, distance_m):
    """Seconds between beeps, or None for silence."""
    if level == 'stop':
        return STOP_GAP_S
    if level in ('notice', 'warning') and distance_m >= 0:
        return distance_m * BEEP_GAP_PER_M
    return None


def bar_value(distance_m):
    """0 to 100, fuller when closer. 0 when nothing is in range."""
    if distance_m < 0 or distance_m >= BAR_FAR_M:
        return 0
    if distance_m <= BAR_NEAR_M:
        return 100
    return round(100 * (BAR_FAR_M - distance_m) / (BAR_FAR_M - BAR_NEAR_M))


def make_beep_file():
    """Write a short 880 Hz beep to a temporary .wav file and return its path."""
    rate = 22050
    count = int(rate * BEEP_S)
    frames = bytearray()
    for i in range(count):
        fade = min(1.0, i / 200, (count - i) / 200)          # avoid clicks at start/end
        sample = int(12000 * fade * math.sin(2 * math.pi * 880 * i / rate))
        frames += sample.to_bytes(2, 'little', signed=True)
    path = os.path.join(tempfile.gettempdir(), 'apnvi_beep.wav')
    with wave.open(path, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(bytes(frames))
    return path


class OutputNode(Node):
    """Speak, beep and draw the bar from what hazard_node sends."""

    def __init__(self):
        super().__init__('output_node')
        self.declare_parameter('show_bar', True)
        self.declare_parameter('voice', False)          # no speech to the user (decided)
        self.show_bar = bool(self.get_parameter('show_bar').value)
        self.voice = bool(self.get_parameter('voice').value)

        self.level, self.status, self.distance_m = 'clear', None, -1.0
        self.old_level, self.old_status = 'clear', None
        self.last_input = None                  # when hazard_node last sent something
        self.last_said = {}                     # kind -> when it was last spoken
        self.speech = None                      # the sentence being played right now
        self.next_beep = 0.0
        self.last_bar = 0.0

        self.espeak = shutil.which('espeak-ng')
        self.player = shutil.which('paplay') or shutil.which('aplay')
        self.beep_file = make_beep_file()
        if not self.espeak:
            self.get_logger().warn('espeak-ng not found: messages will only be printed')
        self.sounds = {}                        # sentence -> ready-made .wav file
        if self.voice:
            self.prepare_common_sentences()
        if not self.player:
            self.get_logger().warn('no sound player (paplay/aplay) found: no beeps')

        self.create_subscription(String, '/hazard/level', self.on_level, 10)
        self.create_subscription(Float32, '/hazard/distance', self.on_distance, 10)
        self.create_subscription(String, '/sensor/status', self.on_status, 10)
        self.create_timer(0.02, self.on_timer)
        self.get_logger().info(f'output_node started, voice {"on" if self.voice else "off"}')

    def on_level(self, msg):
        self.level = msg.data
        self.last_input = time.monotonic()

    def on_distance(self, msg):
        self.distance_m = float(msg.data)
        self.last_input = time.monotonic()

    def on_status(self, msg):
        self.status = msg.data
        self.last_input = time.monotonic()

    def sound_file(self, sentence):
        """Return a .wav file of the sentence, making it with espeak-ng the first time."""
        if sentence in self.sounds:
            return self.sounds[sentence]
        if not (self.espeak and self.player):
            return None
        os.makedirs(SOUND_DIR, exist_ok=True)
        path = os.path.join(SOUND_DIR, f'say_{len(self.sounds)}.wav')
        try:
            subprocess.run([self.espeak, '-s', VOICE_SPEED, '-w', path, sentence],
                           check=True, timeout=5, capture_output=True)
        except Exception as error:              # could not make the file: speak directly
            self.get_logger().warn(f'could not prepare "{sentence}": {error}')
            return None
        self.sounds[sentence] = path
        return path

    def prepare_common_sentences(self):
        """Make the sound files for every usual sentence now, so speaking is instant later."""
        sentences = ['Stop', 'Camera blocked', 'Camera back']
        sentences += [f'Obstacle ahead, {tenths / 10:.1f} metres' for tenths in range(5, 16)]
        for sentence in sentences:
            self.sound_file(sentence)
        self.get_logger().info(f'{len(self.sounds)} sentences ready to play')

    def speaking(self):
        """Return True while a sentence is still playing."""
        return self.speech is not None and self.speech.poll() is None

    def say(self, messages):
        """
        Speak the new messages in one go, e.g. "Camera blocked. Obstacle ahead, 0.6 metres".

        A kind of message spoken less than 4 s ago is skipped.
        """
        if not self.voice:
            return                              # voice is off: beeps only
        now = time.monotonic()
        texts = []
        for kind, text in messages:
            if now - self.last_said.get(kind, -REPEAT_S) >= REPEAT_S:
                self.last_said[kind] = now
                texts.append(text)
        if not texts:
            return
        sentence = '. '.join(texts)
        self.get_logger().info(f'SAY: {sentence}')
        if not self.espeak:
            return
        if self.speaking():
            self.speech.terminate()             # newest message wins (e.g. "Stop")
        path = self.sound_file(sentence)
        command = [self.player, path] if path else [self.espeak, '-s', VOICE_SPEED, sentence]
        self.speech = subprocess.Popen(command,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def beep(self):
        if self.player:
            subprocess.Popen([self.player, self.beep_file],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def on_timer(self):
        """Run 50 times a second: speech on changes, beeps, bar."""
        now = time.monotonic()
        fresh = self.last_input is not None and now - self.last_input < INPUT_TIMEOUT_S
        level = self.level if fresh else 'clear'
        distance_m = self.distance_m if fresh else -1.0

        # 1. Speak when the level or the camera status changes.
        if fresh and self.status is not None:
            self.say(message_for(level, self.status, distance_m,
                                 self.old_level, self.old_status))
            self.old_level, self.old_status = level, self.status

        # 2. Beep, faster when closer, but not while speaking.
        gap = beep_gap(level, distance_m)
        if gap is not None and not self.speaking() and now >= self.next_beep:
            self.beep()
            self.next_beep = now + BEEP_S + gap

        # 3. Bar on screen, 5 times a second.
        if self.show_bar and now - self.last_bar >= 0.2:
            self.last_bar = now
            value = bar_value(distance_m)
            filled = value // 5
            print(f'\r[{"#" * filled}{"-" * (20 - filled)}] {value:3d}  {level:8s} '
                  f'{distance_m:5.2f} m  {self.status or "waiting":15s}', end='', flush=True)


def main(args=None):
    """Start the node and keep it running until Ctrl+C."""
    rclpy.init(args=args)
    node = OutputNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        print()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
