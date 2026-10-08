"""
Tests for finder.py with no model and no campus PC.

The answers below copy the format of Qwen3-VL's own grounding examples. A tiny fake
Ollama server, started inside the test, stands in for the campus PC.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

from apnvi_stage2.finder import draw_box, find, make_prompt, read_boxes, to_pixels
import numpy as np
import pytest

QWEN_STYLE = '```json\n[\n\t{"bbox_2d": [100, 200, 300, 400], "label": "exit sign"}\n]\n```'


def photo(width=640, height=480):
    """Make a plain grey photo."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


# ---- Reading Qwen's answer ----

def test_reads_qwen_style_answer():
    assert read_boxes(QWEN_STYLE) == [(100.0, 200.0, 300.0, 400.0, 'exit sign')]


def test_empty_list_means_not_there():
    assert read_boxes('[]') == []


def test_reads_answer_without_fences_and_with_extra_words():
    answer = 'Here it is: [{"bbox_2d": [1, 2, 3, 4], "label": "ESB2090"}] done.'
    assert read_boxes(answer) == [(1.0, 2.0, 3.0, 4.0, 'ESB2090')]


def test_reads_single_box_without_list():
    assert read_boxes('{"bbox_2d": [5, 6, 7, 8]}') == [(5.0, 6.0, 7.0, 8.0, '')]


def test_ignores_thinking_part():
    answer = '<think>maybe [1, 2]</think>[{"bbox_2d": [10, 20, 30, 40], "label": "WC"}]'
    assert read_boxes(answer) == [(10.0, 20.0, 30.0, 40.0, 'WC')]


@pytest.mark.parametrize('answer', [
    'I cannot see a bathroom.', '[{"label": "door"}]', '[{"bbox_2d": [1, 2, 3]}]', '[1, 2',
])
def test_nonsense_cannot_be_understood(answer):
    assert read_boxes(answer) is None


def test_keeps_all_boxes_in_order():
    answer = '[{"bbox_2d": [1, 1, 2, 2], "label": "a"}, {"bbox_2d": [3, 3, 4, 4], "label": "b"}]'
    assert [box[4] for box in read_boxes(answer)] == ['a', 'b']


# ---- Turning 0-1000 into pixels ----

def test_scales_to_photo_size():
    assert to_pixels((0, 0, 1000, 1000), 640, 480) == (0, 0, 639, 479)
    assert to_pixels((500, 500, 750, 1000), 1001, 1001) == (500, 500, 750, 1000)


def test_keeps_box_inside_photo_and_in_order():
    assert to_pixels((900, 1200, -50, 100), 1001, 1001) == (0, 100, 900, 1000)


# ---- The question ----

def test_prompt_contains_question_and_asks_for_empty_list():
    prompt = make_prompt('  where is ESB2090 ')
    assert '"where is ESB2090"' in prompt
    assert 'reply with []' in prompt


# ---- The whole thing, against a fake Ollama ----

class FakeOllama(BaseHTTPRequestHandler):
    """Answers /api/chat like Ollama does, with whatever answer the test sets."""

    answer = '[]'
    last_request = None

    def do_POST(self):  # noqa: N802 (name fixed by http.server)
        """Reply to one request."""
        length = int(self.headers['Content-Length'])
        FakeOllama.last_request = json.loads(self.rfile.read(length))
        body = json.dumps({'message': {'role': 'assistant', 'content': FakeOllama.answer},
                           'done': True}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Keep the test output quiet."""


@pytest.fixture
def fake_pc():
    """Start the fake Ollama on a free port and give back its address."""
    server = HTTPServer(('127.0.0.1', 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()


def test_find_gives_box_in_pixels(fake_pc):
    FakeOllama.answer = QWEN_STYLE
    result = find(photo(1001, 1001), 'where is the exit', url=fake_pc)
    assert result.found and result.box == (100, 200, 300, 400)
    assert result.label == 'exit sign' and result.error == ''


def test_find_sends_what_ollama_expects(fake_pc):
    FakeOllama.answer = '[]'
    find(photo(), 'where is the exit', url=fake_pc, model='qwen3-vl:4b')
    sent = FakeOllama.last_request
    assert sent['model'] == 'qwen3-vl:4b' and sent['stream'] is False
    assert 'where is the exit' in sent['messages'][0]['content']
    assert len(sent['messages'][0]['images']) == 1


def test_find_not_there(fake_pc):
    FakeOllama.answer = '[]'
    result = find(photo(), 'where is the bathroom', url=fake_pc)
    assert not result.found and result.error == ''


def test_find_with_nonsense_answer_is_not_found_with_error(fake_pc):
    FakeOllama.answer = 'no idea'
    result = find(photo(), 'where is the bathroom', url=fake_pc)
    assert not result.found and 'understand' in result.error


def test_find_with_pc_off_does_not_crash():
    result = find(photo(), 'where is the exit', url='http://127.0.0.1:9', timeout_s=2)
    assert not result.found and 'cannot reach' in result.error


def test_big_photo_is_shrunk_but_box_matches_original(fake_pc):
    FakeOllama.answer = '[{"bbox_2d": [0, 0, 1000, 1000], "label": "wall"}]'
    result = find(photo(2560, 1440), 'where is the wall', url=fake_pc)
    assert result.box == (0, 0, 2559, 1439)


# ---- Drawing ----

def test_draw_box_draws_red_and_leaves_original_alone():
    original = photo()
    drawn = draw_box(original, (100, 100, 200, 200), 'exit sign')
    assert tuple(drawn[100, 150]) == (0, 0, 255)
    assert tuple(original[100, 150]) == (128, 128, 128)
