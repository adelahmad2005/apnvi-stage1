"""
Speech server for the campus PC: a recording goes in, the words come out.

Runs on the campus PC (Windows, RTX 2070), next to Ollama. The Jetson sends the user's
question as a WAV recording and gets back the words as text (decided 2026-10-08:
speech to text runs on the campus PC, because it is faster there).

Uses faster-whisper, a free speech-to-text library built on OpenAI's Whisper models.
Nothing from ROS: this runs on Windows with plain Python.

Start it (in a Command Prompt, in this folder):
  python speech_server.py                     graphics card, "base" model, port 8090
  python speech_server.py --device cpu        if the graphics card libraries fail

What it answers:
  POST /transcribe   body = the WAV file       -> {"text": "where is ESB2090", "seconds": 0.4}
  GET  /health                                 -> {"ok": true, "model": "base", ...}
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import time

# ---- Settings (each can be changed on the command line) ----
PORT = 8090
MODEL = 'base'                          # Whisper model size: tiny, base, small, medium
LANGUAGE = 'en'                         # questions are asked in English
DEVICE = 'cuda'                         # 'cuda' = graphics card, 'cpu' = processor
COMPUTE_TYPE = {'cuda': 'float16', 'cpu': 'int8'}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024     # 10 MB, about 5 minutes of 16 kHz audio
# Example questions given to Whisper as a hint, so room codes come out as "ESB2090"
# rather than "E S B 2 0 9 0".
HINT = 'Where is ESB2090? Where is the bathroom? Where is the exit?'


class WhisperEngine:
    """Wraps faster-whisper: load the model once, then turn recordings into text."""

    def __init__(self, model=MODEL, device=DEVICE, language=LANGUAGE):
        from faster_whisper import WhisperModel     # only needed on the campus PC
        self.name = model
        self.device = device
        self.language = language
        self.model = WhisperModel(model, device=device, compute_type=COMPUTE_TYPE[device])

    def to_text(self, wav_bytes):
        """Turn one recording (the bytes of a WAV file) into text."""
        segments, _ = self.model.transcribe(
            io.BytesIO(wav_bytes), language=self.language, beam_size=5,
            vad_filter=True,                       # skip the silence before and after
            initial_prompt=HINT)
        return ' '.join(segment.text.strip() for segment in segments).strip()


def make_handler(engine):
    """Build the request handler that uses this engine."""
    lock = threading.Lock()                        # one recording at a time on the card

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, body):
            """Send a JSON reply."""
            data = json.dumps(body).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 (name fixed by http.server)
            """Answer /health, so the Jetson can check the server is up."""
            if self.path != '/health':
                self.send_json(404, {'error': 'unknown address'})
                return
            self.send_json(200, {'ok': True, 'model': engine.name,
                                 'device': engine.device, 'language': engine.language})

        def do_POST(self):  # noqa: N802 (name fixed by http.server)
            """Answer /transcribe: the body is the WAV file."""
            if self.path != '/transcribe':
                self.send_json(404, {'error': 'unknown address'})
                return
            length = int(self.headers.get('Content-Length') or 0)
            if not 0 < length <= MAX_UPLOAD_BYTES:
                self.send_json(400, {'error': f'recording must be 1 to {MAX_UPLOAD_BYTES} bytes'})
                return
            wav = self.rfile.read(length)
            start = time.monotonic()
            try:
                with lock:
                    text = engine.to_text(wav)
            except Exception as problem:           # bad file, card error: say so, keep running
                self.send_json(500, {'error': f'could not turn the recording into text: '
                                              f'{problem}'})
                return
            self.send_json(200, {'text': text, 'seconds': round(time.monotonic() - start, 3)})

        def log_message(self, fmt, *args):
            """Print one short line per request."""
            print(f'{self.address_string()} {fmt % args}', flush=True)

    return Handler


def serve(engine, port=PORT, host='0.0.0.0'):
    """Return a server (not started yet) that answers on every network address."""
    return ThreadingHTTPServer((host, port), make_handler(engine))


def main():
    """Load the model and answer requests until Ctrl+C."""
    parser = argparse.ArgumentParser(description='APNVI speech server for the campus PC')
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--device', default=DEVICE, choices=sorted(COMPUTE_TYPE))
    parser.add_argument('--language', default=LANGUAGE)
    args = parser.parse_args()

    print(f'Loading Whisper "{args.model}" on {args.device} '
          '(the first time it downloads the model)...', flush=True)
    engine = WhisperEngine(args.model, args.device, args.language)
    server = serve(engine, args.port)
    print(f'Ready. Listening on port {args.port}. Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
