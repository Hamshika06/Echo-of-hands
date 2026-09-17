#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single source of truth for the gesture vocabulary.

gestures.csv holds the class id, the short label drawn on the video, the phrase
that gets spoken aloud, an emoji for the web UI, and a hint describing how to
form the sign. app.py still reads the plain one-column
keypoint_classifier_label.csv, so sync_label_file() regenerates it from here.
"""
import csv
import os

GESTURES_PATH = 'model/keypoint_classifier/gestures.csv'
LABEL_PATH = 'model/keypoint_classifier/keypoint_classifier_label.csv'
DATASET_PATH = 'model/keypoint_classifier/keypoint.csv'


class Gesture(object):
    def __init__(self, id, label, speech, emoji, hint):
        self.id = int(id)
        self.label = label.strip()
        self.speech = speech.strip()
        self.emoji = emoji.strip()
        self.hint = hint.strip()

    def __repr__(self):
        return '<Gesture {} {!r}>'.format(self.id, self.label)


def load_gestures(path=GESTURES_PATH):
    with open(path, encoding='utf-8-sig', newline='') as f:
        gestures = [Gesture(**row) for row in csv.DictReader(f)]

    gestures.sort(key=lambda g: g.id)
    expected = list(range(len(gestures)))
    if [g.id for g in gestures] != expected:
        raise ValueError(
            'gesture ids must be contiguous starting at 0, got {}'.format(
                [g.id for g in gestures]))
    return gestures


def sync_label_file(gestures, path=LABEL_PATH):
    """Rewrite the one-column label file app.py reads."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for gesture in gestures:
            writer.writerow([gesture.label])


def sample_counts(gestures, path=DATASET_PATH):
    """How many training rows exist per class id."""
    counts = {gesture.id: 0 for gesture in gestures}
    if not os.path.exists(path):
        return counts

    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            class_id = int(row[0])
            if class_id in counts:
                counts[class_id] += 1
    return counts
