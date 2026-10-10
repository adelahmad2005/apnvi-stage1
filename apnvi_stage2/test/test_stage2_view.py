"""Tests for the drawing part of stage2_view.py (no ROS, no camera)."""

from apnvi_stage2.stage2_view import answer_lines, render, SHOW_ANSWER_S, WIDTH
import numpy as np


def photo(colour):
    """Make a 1280 x 720 photo of one colour."""
    return np.full((720, 1280, 3), colour, dtype=np.uint8)


def test_no_camera_yet_still_draws():
    frame = render(None, None, None, 99, 'ready')
    assert frame.shape[1] == WIDTH


def test_shows_live_when_no_answer():
    frame = render(photo(50), None, None, 99, 'ready')
    assert tuple(frame[300, 480]) == (50, 50, 50)


def test_shows_answer_for_a_while_then_live_again():
    live, answer = photo(50), photo(200)
    result = {'question': 'where is the exit', 'found': True, 'label': 'exit sign'}
    assert tuple(render(live, answer, result, 1.0, 'ready')[300, 480]) == (200, 200, 200)
    assert tuple(render(live, answer, result, SHOW_ANSWER_S + 1, 'ready')[300, 480]) == \
        (50, 50, 50)


def test_answer_lines():
    assert answer_lines(None)[0].startswith('No question yet')
    found = answer_lines({'question': 'where is the exit', 'found': True,
                          'label': 'exit sign', 'seconds': 1.25})
    assert found == ('Asked: "where is the exit"', 'FOUND: exit sign in 1.2 s (silent)')
    missed = answer_lines({'question': 'where is it', 'found': False, 'seconds': 2.0,
                           'said': "I can't see it, turn slowly and try again"})
    assert missed[1].startswith('NOT FOUND in 2.0 s')
