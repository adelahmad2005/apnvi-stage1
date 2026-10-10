#!/usr/bin/env python3
"""
Try the finder on a whole folder of photos and save copies with the red box drawn on.

For Monday: run it on Mo'ath's test photos once Qwen is running on the campus PC.
Needs only Python with numpy and OpenCV. No ROS.

The question for each photo comes from its file name, the way Mo'ath names them:
  ESB2090_4m.png      -> "where is ESB2090"
  bathroom_sign_2m.jpg -> "where is bathroom sign"
Or put a questions.txt in the folder, one line per photo:  photo.png | where is the exit

Usage (from the repo root):
  python3 tools/try_finder.py PHOTO_FOLDER --url http://CAMPUS_PC_IP:11434
Results: PHOTO_FOLDER/results/ (boxed copies + results.csv) and a table on screen.
"""

import argparse
import csv
import os
import re
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apnvi_stage2'))
from apnvi_stage2.finder import draw_box, find, MODEL, OLLAMA_URL  # noqa: E402

PHOTO_TYPES = ('.png', '.jpg', '.jpeg', '.bmp')


def question_from_name(file_name):
    """Make a question from a file name: 'ESB2090_4m.png' -> 'where is ESB2090'."""
    stem = os.path.splitext(file_name)[0]
    stem = re.sub(r'_\d+(\.\d+)?m$', '', stem)          # drop the distance at the end
    return 'where is ' + stem.replace('_', ' ').strip()


def read_questions(folder):
    """Read questions.txt if there is one: {photo name: question}."""
    path = os.path.join(folder, 'questions.txt')
    questions = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as lines:
            for line in lines:
                if '|' in line:
                    name, question = line.split('|', 1)
                    questions[name.strip()] = question.strip()
    return questions


def main():
    """Run every photo in the folder through the finder."""
    parser = argparse.ArgumentParser(description='Try the Stage 2 finder on a folder')
    parser.add_argument('folder')
    parser.add_argument('--url', default=OLLAMA_URL, help='Ollama address on the campus PC')
    parser.add_argument('--model', default=MODEL)
    args = parser.parse_args()

    out_folder = os.path.join(args.folder, 'results')
    os.makedirs(out_folder, exist_ok=True)
    questions = read_questions(args.folder)
    photos = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(PHOTO_TYPES))
    if not photos:
        sys.exit(f'No photos in {args.folder}')

    rows = []
    for name in photos:
        image = cv2.imread(os.path.join(args.folder, name))
        if image is None:
            print(f'{name}: cannot open, skipped')
            continue
        question = questions.get(name, question_from_name(name))
        result = find(image, question, url=args.url, model=args.model)
        if result.found:
            image = draw_box(image, result.box, result.label)
        cv2.imwrite(os.path.join(out_folder, name), image)
        found = 'yes' if result.found else 'no'
        rows.append([name, question, found, result.label, f'{result.seconds:.1f}',
                     result.error, result.answer.replace('\n', ' ')])
        print(f'{name:28} {question:32} found={found:3} {result.seconds:5.1f} s '
              f'{result.label} {result.error}')

    with open(os.path.join(out_folder, 'results.csv'), 'w', newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['photo', 'question', 'found', 'label', 'seconds', 'error',
                         'qwen answer'])
        writer.writerows(rows)
    print(f'\nBoxed copies and results.csv are in {out_folder}. '
          'Open the photos and check each box by eye.')


if __name__ == '__main__':
    main()
