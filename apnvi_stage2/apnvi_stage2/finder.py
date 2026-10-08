"""
The finder: one colour photo and one question in, a box (or "not found") out.

Plain Python, numpy and OpenCV only. Nothing from ROS, so it can be tested on its own,
the same way as Stage 1's hazard_check.py. The Stage 2 ROS program calls find().

It asks Qwen3-VL-4B, which runs under Ollama on the campus PC (decided 2026-10-08).
Ollama is a program that runs AI models and answers requests over the network.

How Qwen answers (from Qwen's own grounding examples): a short JSON list such as
  [{"bbox_2d": [x1, y1, x2, y2], "label": "exit sign"}]
where the numbers run from 0 to 1000 across the photo, whatever its real size
(0 = left or top edge, 1000 = right or bottom edge). We ask for [] when it is not there.
"""

import base64
from dataclasses import dataclass
import json
import re
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

# ---- Settings ----
OLLAMA_URL = 'http://localhost:11434'   # campus PC address goes here on Monday
MODEL = 'qwen3-vl:4b'                   # the Ollama name of Qwen3-VL-4B
TIMEOUT_S = 30.0                        # give up waiting for an answer after this
KEEP_LOADED = '60m'                     # keep the model in memory between questions
MAX_SIDE_PX = 1280                      # shrink bigger photos first: faster, same box
JPEG_QUALITY = 90
BOX_SCALE = 1000                        # Qwen's box numbers run from 0 to this

BOX_COLOUR = (0, 0, 255)                # red, in OpenCV's blue-green-red order
BOX_THICKNESS = 4

PROMPT = (
    'A blind person asked: "{question}". '
    'Find the one thing in this photo that answers the question. It may be a sign, a '
    'room number, a symbol, a door or an object. '
    'Report its bbox coordinates in JSON format as '
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "short name"}}]. '
    'If it is not in the photo, reply with [] and nothing else.'
)


@dataclass
class FindResult:
    """What find() gives back."""

    found: bool                 # True = there is a box
    box: tuple = None           # (left, top, right, bottom) in pixels of the photo given
    label: str = ''             # what Qwen called it, e.g. 'exit sign'
    seconds: float = 0.0        # how long the question took
    error: str = ''             # filled in when something went wrong (PC not reachable...)
    answer: str = ''            # Qwen's raw answer, for checking by eye


def make_prompt(question):
    """Wrap the user's question in the instruction Qwen gets."""
    return PROMPT.format(question=question.strip().replace('"', "'"))


def photo_to_base64(image_bgr):
    """Shrink the photo if it is big, turn it into a JPEG and then into text for sending."""
    height, width = image_bgr.shape[:2]
    scale = MAX_SIDE_PX / max(height, width)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (round(width * scale), round(height * scale)),
                               interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError('could not turn the photo into a JPEG')
    return base64.b64encode(jpeg.tobytes()).decode('ascii')


def ask_qwen(image_bgr, question, url=OLLAMA_URL, model=MODEL, timeout_s=TIMEOUT_S):
    """Send the photo and the question to Ollama and return Qwen's answer as text."""
    request = {
        'model': model,
        'messages': [{'role': 'user', 'content': make_prompt(question),
                      'images': [photo_to_base64(image_bgr)]}],
        'stream': False,                     # one complete answer, not word by word
        'keep_alive': KEEP_LOADED,
        'options': {'temperature': 0},       # same photo + question = same answer
    }
    data = json.dumps(request).encode('utf-8')
    http = urllib.request.Request(url.rstrip('/') + '/api/chat', data=data,
                                  headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(http, timeout=timeout_s) as response:
        reply = json.loads(response.read().decode('utf-8'))
    return reply['message']['content']


def read_boxes(answer):
    """
    Pull the boxes out of Qwen's answer.

    Returns a list of (x1, y1, x2, y2, label) on Qwen's 0-1000 scale, [] for "not
    there", or None if the answer cannot be understood at all.
    """
    text = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)   # thinking, if any
    fenced = re.search(r'```(?:json)?\s*(.*?)```', text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = min([i for i in (text.find('['), text.find('{')) if i >= 0], default=-1)
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):                 # a single box without the list around it
        data = [data]
    if not isinstance(data, list):
        return None
    boxes = []
    for item in data:
        numbers = item.get('bbox_2d') if isinstance(item, dict) else None
        if not isinstance(numbers, (list, tuple)) or len(numbers) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(n) for n in numbers)
        except (TypeError, ValueError):
            continue
        boxes.append((x1, y1, x2, y2, str(item.get('label', ''))))
    if data and not boxes:
        return None                            # there was something, but not a box
    return boxes


def to_pixels(box, width, height):
    """Turn one box on Qwen's 0-1000 scale into whole pixels of a width x height photo."""
    x1, y1, x2, y2 = box[:4]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    def scale(value, size):
        """Scale one number and keep it inside the photo."""
        return int(round(min(max(value, 0.0), BOX_SCALE) / BOX_SCALE * (size - 1)))

    return scale(left, width), scale(top, height), scale(right, width), scale(bottom, height)


def find(image_bgr, question, url=OLLAMA_URL, model=MODEL, timeout_s=TIMEOUT_S):
    """
    Ask where the thing in the question is, in this photo.

    Never raises: problems come back as FindResult(found=False, error=...).
    If Qwen gives several boxes, the first one is used.
    """
    start = time.monotonic()
    image = np.asarray(image_bgr)
    try:
        answer = ask_qwen(image, question, url, model, timeout_s)
    except (urllib.error.URLError, TimeoutError, OSError) as problem:
        return FindResult(False, seconds=time.monotonic() - start,
                          error=f'cannot reach Qwen at {url}: {problem}')
    except (KeyError, ValueError) as problem:
        return FindResult(False, seconds=time.monotonic() - start,
                          error=f'unexpected reply from Ollama: {problem}')
    seconds = time.monotonic() - start
    boxes = read_boxes(answer)
    if boxes is None:
        return FindResult(False, seconds=seconds, answer=answer,
                          error='could not understand the answer')
    if not boxes:
        return FindResult(False, seconds=seconds, answer=answer)
    height, width = image.shape[:2]
    first = boxes[0]
    return FindResult(True, to_pixels(first, width, height), first[4], seconds, '', answer)


def draw_box(image_bgr, box, label=''):
    """Return a copy of the photo with a red box (and its label) drawn on it."""
    picture = np.array(image_bgr, copy=True)
    left, top, right, bottom = box
    cv2.rectangle(picture, (left, top), (right, bottom), BOX_COLOUR, BOX_THICKNESS)
    if label:
        y = top - 10 if top > 30 else bottom + 30
        cv2.putText(picture, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    BOX_COLOUR, 2, cv2.LINE_AA)
    return picture
