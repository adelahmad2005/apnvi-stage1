"""
Tests for speech_server.py with a fake speech engine (no faster-whisper, no graphics card).

Run from the repo root:  python3 -m pytest campus_pc
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from speech_server import serve  # noqa: E402 (needs the path line above)


class FakeEngine:
    """Stands in for Whisper: returns a fixed sentence, or fails when told to."""

    name, device, language = 'fake', 'cpu', 'en'

    def __init__(self, text='where is ESB2090', fail=False):
        self.text, self.fail, self.received = text, fail, None

    def to_text(self, wav_bytes):
        """Pretend to turn the recording into text."""
        self.received = wav_bytes
        if self.fail:
            raise RuntimeError('broken recording')
        return self.text


@pytest.fixture
def start():
    """Start a speech server on a free port with the given engine; stop it afterwards."""
    servers = []

    def run(engine):
        server = serve(engine, port=0, host='127.0.0.1')
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f'http://127.0.0.1:{server.server_port}'

    yield run
    for server in servers:
        server.shutdown()
        server.server_close()


def post(url, body):
    """Send body to url and return (status, reply as a dict)."""
    request = urllib.request.Request(url, data=body, headers={'Content-Type': 'audio/wav'})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as problem:
        return problem.code, json.loads(problem.read())


def test_recording_comes_back_as_text(start):
    engine = FakeEngine()
    status, reply = post(start(engine) + '/transcribe', b'RIFF fake wav')
    assert status == 200 and reply['text'] == 'where is ESB2090'
    assert engine.received == b'RIFF fake wav'
    assert reply['seconds'] >= 0


def test_health(start):
    with urllib.request.urlopen(start(FakeEngine()) + '/health', timeout=5) as response:
        assert json.loads(response.read())['ok'] is True


def test_empty_recording_is_refused(start):
    status, reply = post(start(FakeEngine()) + '/transcribe', b'')
    assert status == 400 and 'error' in reply


def test_engine_failure_is_reported_and_server_keeps_running(start):
    url = start(FakeEngine(fail=True))
    status, reply = post(url + '/transcribe', b'x')
    assert status == 500 and 'broken recording' in reply['error']
    with urllib.request.urlopen(url + '/health', timeout=5) as response:
        assert response.status == 200


def test_unknown_address(start):
    status, _ = post(start(FakeEngine()) + '/somewhere', b'x')
    assert status == 404
