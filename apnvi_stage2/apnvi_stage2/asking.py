"""
What happens to one question, from the recording (or typed words) to the answer.

Plain Python, nothing from ROS, so it can be tested with fake helpers. stage2_node.py
calls answer_recording() or answer_typed() and then shows or says what comes back.

Decided 2026-10-08 (stage-design.md, Stage 2 section 10):
  found      -> red box on the screen, silent
  not found  -> "I can't see it, turn slowly and try again"
The other three sentences were decided by Adel on 2026-10-10.
"""

from dataclasses import dataclass

import numpy as np

from .finder import draw_box, find, FindResult
from .speech import to_text

NOT_FOUND = "I can't see it, turn slowly and try again"          # thing not in view
NOT_HEARD = "I didn't catch that. Hold the button and ask again"   # no words heard
NO_HELPER = "I can't reach the helper computer"                    # campus PC unreachable
NO_CAMERA = 'The camera is not ready'                              # no picture yet


@dataclass
class Outcome:
    """The answer to one question."""

    question: str = ''          # the words, as heard or typed
    say: str = ''               # sentence to speak; '' = stay silent
    result: FindResult = None   # the finder's answer, None if it was never asked
    picture: object = None      # the photo, with the red box drawn on when found
    problem: str = ''           # for the log: what went wrong, if anything


def answer_typed(question, image_bgr, ollama_url, model, finder=find):
    """Answer a question given as text (typed, or already turned into words)."""
    question = question.strip()
    if not question:
        return Outcome(say=NOT_HEARD, problem='empty question')
    if image_bgr is None:
        return Outcome(question, NO_CAMERA, problem='no colour picture from the camera yet')
    image = np.asarray(image_bgr)
    result = finder(image, question, url=ollama_url, model=model)
    if result.found:
        return Outcome(question, '', result, draw_box(image, result.box, result.label))
    if result.error.startswith(('cannot reach', 'unexpected reply')):
        return Outcome(question, NO_HELPER, result, image, result.error)
    # Not there, or an answer that made no sense: either way the user hears "not found".
    return Outcome(question, NOT_FOUND, result, image, result.error)


def answer_recording(wav_bytes, image_bgr, speech_url, ollama_url, model,
                     speech=to_text, finder=find):
    """Answer a spoken question: turn the recording into words, then answer those."""
    heard = speech(wav_bytes, url=speech_url)
    if heard.error:
        return Outcome(say=NO_HELPER, problem=heard.error)
    if not heard.text:
        return Outcome(say=NOT_HEARD, problem='no words in the recording')
    return answer_typed(heard.text, image_bgr, ollama_url, model, finder)
