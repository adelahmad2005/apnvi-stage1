"""Tests for the no-ROS parts of esp32_reader.py and output_node.py."""

from apnvi_stage1.esp32_reader import fake_distance, parse_line
from apnvi_stage1.output_node import bar_value, beep_gap, message_for


# ---- ESP32 reader ----

def test_parse_numbers_and_skip_junk():
    assert parse_line('1.23\r') == 1.23
    assert parse_line('-1') == -1.0
    assert parse_line('') is None
    assert parse_line('hello') is None


def test_test_mode_goes_3m_to_03m_and_back():
    assert fake_distance(0) == 3.0
    assert fake_distance(9) == 0.3                  # 2.7 m at 0.3 m/s
    assert fake_distance(18) == 3.0                 # and back
    assert all(0.3 <= fake_distance(t / 10) <= 3.0 for t in range(400))


# ---- What gets said ----

def test_warning_says_distance_with_one_decimal():
    assert message_for('warning', 'camera_ok', 1.26, 'notice', 'camera_ok') == \
        [('warning', 'Obstacle ahead, 1.3 metres')]


def test_stop_says_stop():
    assert message_for('stop', 'camera_ok', 0.4, 'warning', 'camera_ok') == [('stop', 'Stop')]


def test_camera_blocked_and_back():
    assert message_for('clear', 'camera_blocked', -1, 'clear', 'camera_ok') == \
        [('blocked', 'Camera blocked')]
    assert message_for('clear', 'camera_ok', 3.0, 'clear', 'camera_blocked') == \
        [('back', 'Camera back')]


def test_no_change_says_nothing():
    assert message_for('warning', 'camera_ok', 1.0, 'warning', 'camera_ok') == []
    assert message_for('notice', 'camera_ok', 1.8, 'clear', 'camera_ok') == []   # beeps only


def test_first_camera_ok_is_not_camera_back():
    assert message_for('clear', 'camera_ok', 3.0, 'clear', None) == []


# ---- Beeps and bar ----

def test_beep_gap():
    assert beep_gap('clear', 3.0) is None
    assert beep_gap('notice', 2.0) == 1.0
    assert beep_gap('warning', 1.0) == 0.5
    assert beep_gap('stop', 0.3) == 0.05


def test_bar_fuller_when_closer():
    assert bar_value(-1) == 0
    assert bar_value(3.5) == 0
    assert bar_value(0.2) == 100
    assert bar_value(1.65) == 50
