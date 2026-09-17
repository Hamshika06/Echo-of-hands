#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Guided training-data collector for the gesture vocabulary.

app.py's built-in logging mode maps one class per number key, which stops at
ten classes. This walks through gestures.csv instead, so any number of classes
can be recorded, and captures a timed burst of samples per sign rather than
relying on the user to hold a key down.

Controls
    n / p        next / previous gesture
    SPACE        record a burst for the selected gesture
    u            undo the most recent burst
    ESC          quit
"""
import argparse
import copy
import csv
import os

import cv2 as cv
import mediapipe as mp

from app import calc_bounding_rect, calc_landmark_list, draw_landmarks, \
    pre_process_landmark
from utils import load_gestures, sample_counts, sync_label_file
from utils.gestures import DATASET_PATH


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=60,
                        help='samples captured per burst')
    parser.add_argument("--countdown", type=float, default=3.0,
                        help='seconds to get your hand in position')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def append_rows(rows):
    with open(DATASET_PATH, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def drop_last_rows(count):
    """Remove the trailing `count` rows from the dataset file."""
    if count <= 0 or not os.path.exists(DATASET_PATH):
        return 0

    with open(DATASET_PATH, newline='') as f:
        rows = [row for row in csv.reader(f) if row]

    removed = min(count, len(rows))
    with open(DATASET_PATH, 'w', newline='') as f:
        csv.writer(f).writerows(rows[:len(rows) - removed])
    return removed


def draw_panel(image, gesture, counts, index, total, status, status_color,
               jump='', last_saved=''):
    overlay = image.copy()
    cv.rectangle(overlay, (0, 0), (image.shape[1], 118), (0, 0, 0), -1)
    cv.rectangle(overlay, (0, image.shape[0] - 34),
                 (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    cv.putText(image, "[{}/{}]  {}".format(index + 1, total, gesture.label),
               (12, 38), cv.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2,
               cv.LINE_AA)
    cv.putText(image, "samples: {}".format(counts[gesture.id]), (12, 68),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1, cv.LINE_AA)
    cv.putText(image, gesture.hint[:70], (12, 94), cv.FONT_HERSHEY_SIMPLEX,
               0.55, (200, 200, 200), 1, cv.LINE_AA)
    cv.putText(image, status, (12, image.shape[0] - 11),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv.LINE_AA)

    # Which class the samples will actually land in, stated plainly, because a
    # burst recorded against the wrong sign is invisible until training.
    target = 'SAVES TO -> [{}] {}'.format(gesture.id, gesture.label)
    size = cv.getTextSize(target, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv.putText(image, target, (image.shape[1] - size[0] - 12, 38),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 255), 2, cv.LINE_AA)

    if jump:
        cv.putText(image, 'jump to: {}_  (ENTER)'.format(jump),
                   (image.shape[1] - 240, 68), cv.FONT_HERSHEY_SIMPLEX, 0.55,
                   (255, 220, 120), 1, cv.LINE_AA)
    elif last_saved:
        cv.putText(image, 'last saved: {}'.format(last_saved[:22]),
                   (image.shape[1] - 240, 68), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (160, 230, 160), 1, cv.LINE_AA)
    return image


def main():
    args = get_args()

    gestures = load_gestures()
    sync_label_file(gestures)
    counts = sample_counts(gestures)

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit('could not open camera device {}'.format(args.device))

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    index = 0
    recording = False
    countdown_frames = 0
    pending = []
    last_burst = 0
    last_saved = ''
    jump = ''

    # Accept upper and lower case, and the bracket keys, so a stray Caps Lock
    # or Shift does not silently swallow the navigation keys.
    NEXT_KEYS = {ord('n'), ord('N'), ord('.'), ord(']')}
    PREV_KEYS = {ord('p'), ord('P'), ord(','), ord('[')}
    UNDO_KEYS = {ord('u'), ord('U')}

    print('Recording to {}'.format(DATASET_PATH))
    print('n/p = change sign, digits+ENTER = jump, SPACE = record, '
          'u = undo, ESC = quit')

    while True:
        key = cv.waitKey(10) & 0xFF
        if key == 27:
            break
        elif key in NEXT_KEYS:
            index = (index + 1) % len(gestures)
            recording, jump = False, ''
        elif key in PREV_KEYS:
            index = (index - 1) % len(gestures)
            recording, jump = False, ''
        elif key in UNDO_KEYS:
            removed = drop_last_rows(last_burst)
            if removed:
                counts = sample_counts(gestures)
                print('undo: removed {} samples from {!r}'.format(
                    removed, last_saved))
                last_burst = 0
                last_saved = ''
        elif 48 <= key <= 57:  # build a class id, e.g. 1 then 7 -> 17
            jump = (jump + chr(key))[-2:]
        elif key in (8, 127):  # backspace
            jump = jump[:-1]
        elif key in (13, 10):  # enter: jump to the typed class id
            if jump and int(jump) < len(gestures):
                index = int(jump)
                recording = False
            jump = ''
        elif key == 32 and not recording:
            recording = True
            pending = []
            countdown_frames = int(args.countdown * 30)

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)

        gesture = gestures[index]
        landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = calc_landmark_list(debug_image, hand_landmarks)
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            cv.rectangle(debug_image, (brect[0], brect[1]),
                         (brect[2], brect[3]), (0, 0, 0), 1)
            debug_image = draw_landmarks(debug_image, landmarks)

        if recording and countdown_frames > 0:
            countdown_frames -= 1
            seconds = countdown_frames / 30.0 + 1
            status = 'GET READY... {:.0f}'.format(seconds)
            colour = (0, 200, 255)
        elif recording:
            if landmarks is not None:
                pending.append([gesture.id, *pre_process_landmark(landmarks)])
            status = 'RECORDING {}/{}  (hold the sign)'.format(
                len(pending), args.samples)
            colour = (0, 80, 255)

            if len(pending) >= args.samples:
                append_rows(pending)
                counts[gesture.id] += len(pending)
                last_burst = len(pending)
                last_saved = gesture.label
                print('saved {} samples for [{}] {!r}  (total {})'.format(
                    len(pending), gesture.id, gesture.label,
                    counts[gesture.id]))
                recording = False
                pending = []
        else:
            if landmarks is None:
                status = 'no hand detected - SPACE to record'
                colour = (120, 120, 255)
            else:
                status = 'hand ready - SPACE to record'
                colour = (150, 255, 150)

        debug_image = draw_panel(debug_image, gesture, counts, index,
                                 len(gestures), status, colour, jump,
                                 last_saved)
        cv.imshow('Collect Gesture Data', debug_image)

    cap.release()
    cv.destroyAllWindows()

    print('\nSample counts:')
    for gesture in gestures:
        print('  {:>2}  {:<12} {}'.format(gesture.id, gesture.label,
                                          counts[gesture.id]))


if __name__ == '__main__':
    main()
