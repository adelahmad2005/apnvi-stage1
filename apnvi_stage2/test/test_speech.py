"""Tests for speech.py against a tiny fake speech server (no campus PC needed)."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

from apnvi_stage2.speech import to_text
import pytest


class FakeSpeechServer(BaseHTTPRequestHandler):
    """Answers /transcribe like campus_pc/speech_server.py does."""

    status = 200
    reply = {'text': ' where is ESB2090 ', 'seconds': 0.1}
    received = None

    def do_POST(self):  # noqa: N802 (name fixed by http.server)
        """Reply to one recording."""
        FakeSpeechServer.received = self.rfile.read(int(self.headers['Content-Length']))
        body = json.dumps(FakeSpeechServer.reply).encode()
        self.send_response(FakeSpeechServer.status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Keep the test output quiet."""


@pytest.fixture
def fake_pc():
    """Start the fake server on a free port and give back its address."""
    FakeSpeechServer.status = 200
    FakeSpeechServer.reply = {'text': ' where is ESB2090 ', 'seconds': 0.1}
    server = HTTPServer(('127.0.0.1', 0), FakeSpeechServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()


def test_words_come_back(fake_pc):
    result = to_text(b'RIFF fake wav', url=fake_pc)
    assert result.text == 'where is ESB2090' and result.error == ''
    assert FakeSpeechServer.received == b'RIFF fake wav'


def test_nothing_heard_is_empty_text(fake_pc):
    FakeSpeechServer.reply = {'text': '', 'seconds': 0.1}
    result = to_text(b'x', url=fake_pc)
    assert result.text == '' and result.error == ''


def test_server_error_is_reported(fake_pc):
    FakeSpeechServer.status = 500
    FakeSpeechServer.reply = {'error': 'broken recording'}
    result = to_text(b'x', url=fake_pc)
    assert result.text == '' and 'broken recording' in result.error


def test_pc_off_does_not_crash():
    result = to_text(b'x', url='http://127.0.0.1:9', timeout_s=2)
    assert result.text == '' and 'cannot reach' in result.error
