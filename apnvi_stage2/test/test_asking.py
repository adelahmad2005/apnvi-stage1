"""
Tests for asking.py and the no-ROS helpers in stage2_node.py, with fake helpers.

No campus PC, no microphone, no camera. A fake 'arecord' stands in for the microphone.
"""

import json
import os
import stat
import time

from apnvi_stage2.asking import (
    answer_recording, answer_typed, NO_CAMERA, NO_HELPER, NOT_FOUND, NOT_HEARD)
from apnvi_stage2.finder import FindResult
from apnvi_stage2.speech import SpeechResult
from apnvi_stage2.stage2_node import bgr_to_image, image_to_bgr, Recorder, result_json
import numpy as np
from sensor_msgs.msg import Image

URL, MODEL = 'http://pc:11434', 'qwen3-vl:4b'


def photo():
    """Make a plain grey photo."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


def finder_saying(result):
    """Make a fake finder that always gives this result and remembers the question."""
    def fake(image, question, url, model):
        fake.asked = question
        return result
    fake.asked = None
    return fake


def speech_saying(text='', error=''):
    """Make a fake speech-to-text that always gives these words."""
    return lambda wav, url: SpeechResult(text, 0.2, error)


# ---- Typed questions ----

def test_found_is_silent_with_red_box():
    outcome = answer_typed('where is the exit', photo(), URL, MODEL,
                           finder_saying(FindResult(True, (10, 10, 50, 50), 'exit sign')))
    assert outcome.say == ''
    assert tuple(outcome.picture[10, 30]) == (0, 0, 255)


def test_not_found_says_turn_slowly():
    outcome = answer_typed('where is the exit', photo(), URL, MODEL,
                           finder_saying(FindResult(False)))
    assert outcome.say == NOT_FOUND == "I can't see it, turn slowly and try again"


def test_nonsense_answer_counts_as_not_found():
    result = FindResult(False, error='could not understand the answer')
    assert answer_typed('where is it', photo(), URL, MODEL, finder_saying(result)).say == NOT_FOUND


def test_pc_unreachable_says_so():
    result = FindResult(False, error='cannot reach Qwen at http://pc:11434: refused')
    assert answer_typed('where is it', photo(), URL, MODEL, finder_saying(result)).say == NO_HELPER


def test_no_camera_picture_yet():
    finder = finder_saying(FindResult(True, (1, 1, 2, 2)))
    assert answer_typed('where is it', None, URL, MODEL, finder).say == NO_CAMERA
    assert finder.asked is None                     # never bothered the PC


def test_empty_typed_question():
    assert answer_typed('   ', photo(), URL, MODEL, finder_saying(FindResult(True))).say == \
        NOT_HEARD


# ---- Spoken questions ----

def test_spoken_question_is_passed_to_finder():
    finder = finder_saying(FindResult(False))
    outcome = answer_recording(b'wav', photo(), 'http://pc:8090', URL, MODEL,
                               speech_saying('where is ESB2090'), finder)
    assert finder.asked == 'where is ESB2090' and outcome.question == 'where is ESB2090'


def test_nothing_heard():
    outcome = answer_recording(b'wav', photo(), 'http://pc:8090', URL, MODEL,
                               speech_saying(''), finder_saying(FindResult(True)))
    assert outcome.say == NOT_HEARD


def test_speech_server_down():
    outcome = answer_recording(b'wav', photo(), 'http://pc:8090', URL, MODEL,
                               speech_saying(error='cannot reach the speech server'),
                               finder_saying(FindResult(True)))
    assert outcome.say == NO_HELPER


# ---- Pictures in and out of ROS ----

def test_rgb_picture_becomes_bgr_and_back():
    msg = Image()
    msg.height, msg.width, msg.encoding = 2, 3, 'rgb8'
    msg.step = 3 * 3 + 4                           # rows may have padding at the end
    rows = [bytes([255, 0, 0] * 3) + b'\0' * 4] * 2   # pure red, in red-green-blue order
    msg.data = b''.join(rows)
    picture = image_to_bgr(msg)
    assert picture.shape == (2, 3, 3) and tuple(picture[0, 0]) == (0, 0, 255)
    back = bgr_to_image(picture)
    assert back.encoding == 'bgr8' and back.step == 9
    assert bytes(back.data)[:3] == bytes([0, 0, 255])


def test_result_json():
    from apnvi_stage2.asking import Outcome
    outcome = Outcome('where is the exit', '', FindResult(True, (1, 2, 3, 4), 'exit', 1.234))
    data = json.loads(result_json(outcome))
    assert data['found'] is True and data['box'] == [1, 2, 3, 4] and data['seconds'] == 1.23


# ---- Recording, with a fake arecord ----

FAKE_ARECORD = """#!/usr/bin/env python3
import signal, sys, time
path = sys.argv[-1]
with open(path, 'wb') as f:
    f.write(b'RIFF' + b'\\0' * 40)                  # 44-byte header
def finish(*_):
    with open(path, 'ab') as f:
        f.write(b'sound' * 100)
    sys.exit(0)
signal.signal(signal.SIGINT, finish)
while True:
    time.sleep(0.05)
"""


def test_recorder_gets_the_sound_after_letting_go(tmp_path, monkeypatch):
    fake = tmp_path / 'arecord'
    fake.write_text(FAKE_ARECORD)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', f'{tmp_path}{os.pathsep}{os.environ["PATH"]}')
    recorder = Recorder()
    recorder.start()
    time.sleep(0.5)
    path = recorder.path
    wav, held = recorder.stop()
    assert wav is not None and wav.endswith(b'sound') and held >= 0.4
    assert not os.path.exists(path)                 # the temporary file is cleaned up
