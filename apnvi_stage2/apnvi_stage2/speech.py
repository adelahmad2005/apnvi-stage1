"""
Speech to text, Jetson side: send a recording to the campus PC, get the words back.

Plain Python, nothing from ROS. The other half is campus_pc/speech_server.py, which
runs on the campus PC (decided 2026-10-08: speech to text runs there, because it is
faster).
"""

from dataclasses import dataclass
import json
import time
import urllib.error
import urllib.request

SPEECH_URL = 'http://localhost:8090'    # campus PC address goes here on Monday
TIMEOUT_S = 15.0                        # give up waiting for the words after this


@dataclass
class SpeechResult:
    """What to_text() gives back."""

    text: str = ''              # the words, e.g. 'where is ESB2090'. '' = nothing heard
    seconds: float = 0.0        # how long the round trip took
    error: str = ''             # filled in when something went wrong


def to_text(wav_bytes, url=SPEECH_URL, timeout_s=TIMEOUT_S):
    """
    Send one WAV recording to the speech server and return the words.

    Never raises: problems come back as SpeechResult(error=...).
    """
    start = time.monotonic()
    request = urllib.request.Request(url.rstrip('/') + '/transcribe', data=wav_bytes,
                                     headers={'Content-Type': 'audio/wav'})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            reply = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as problem:          # the server answered with an error
        try:
            detail = json.loads(problem.read().decode('utf-8')).get('error', '')
        except (ValueError, AttributeError):
            detail = ''
        return SpeechResult(seconds=time.monotonic() - start,
                            error=f'speech server error {problem.code}: {detail}')
    except (urllib.error.URLError, TimeoutError, OSError) as problem:
        return SpeechResult(seconds=time.monotonic() - start,
                            error=f'cannot reach the speech server at {url}: {problem}')
    except ValueError as problem:
        return SpeechResult(seconds=time.monotonic() - start,
                            error=f'unexpected reply from the speech server: {problem}')
    return SpeechResult(str(reply.get('text', '')).strip(), time.monotonic() - start)
