#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retrain the hand sign classifier from keypoint.csv.

The notebook version hardcodes the class count and only ever reports training
accuracy. This reads the class count from gestures.csv, holds out a stratified
test split so the reported score means something, and exports the weights as
JSON so the web app can run the same model in the browser.
"""
import argparse
import csv
import json
import os

import numpy as np
import tensorflow as tf

from utils import load_gestures, sync_label_file
from utils.gestures import DATASET_PATH

MODEL_DIR = 'model/keypoint_classifier'
HDF5_PATH = os.path.join(MODEL_DIR, 'keypoint_classifier.keras')
TFLITE_PATH = os.path.join(MODEL_DIR, 'keypoint_classifier.tflite')
WEB_WEIGHTS_PATH = 'web/model_weights.json'

RANDOM_SEED = 42


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_size', type=float, default=0.25)
    return parser.parse_args()


def load_dataset(num_classes):
    features, labels = [], []
    with open(DATASET_PATH, newline='') as f:
        for line_no, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            class_id = int(row[0])
            values = [float(v) for v in row[1:]]
            if len(values) != 42:
                raise ValueError(
                    'row {} has {} features, expected 42'.format(
                        line_no, len(values)))
            if not 0 <= class_id < num_classes:
                raise ValueError(
                    'row {} has class id {} outside 0..{}'.format(
                        line_no, class_id, num_classes - 1))
            features.append(values)
            labels.append(class_id)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def stratified_split(X, y, test_size, seed=RANDOM_SEED):
    """Per-class split, so rare classes still appear in both halves."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for class_id in np.unique(y):
        idx = np.flatnonzero(y == class_id)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        n_test = min(max(n_test, 1), len(idx) - 1) if len(idx) > 1 else 0
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def build_model(num_classes):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])


def print_confusion(y_true, y_pred, gestures):
    num_classes = len(gestures)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1

    present = [g for g in gestures if matrix[g.id].sum() > 0]
    width = max(len(g.label) for g in present) + 1

    print('\nConfusion matrix (rows = actual, columns = predicted)')
    header = ' ' * (width + 2) + ' '.join(
        '{:>4}'.format(g.id) for g in present)
    print(header)
    for gesture in present:
        row = matrix[gesture.id]
        cells = ' '.join('{:>4}'.format(row[g.id]) for g in present)
        total = row.sum()
        hits = row[gesture.id]
        print('{:>2} {:<{w}} {}   {:>5.1f}%'.format(
            gesture.id, gesture.label, cells, 100.0 * hits / total, w=width))


def export_web_weights(model, gestures, counts, path=WEB_WEIGHTS_PATH):
    """Dump dense layer weights so the browser can run the same network."""
    layers = []
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            continue
        weights, biases = layer.get_weights()
        layers.append({
            'activation': layer.activation.__name__,
            'weights': weights.tolist(),
            'biases': biases.tolist(),
        })

    payload = {
        'input_size': 42,
        'layers': layers,
        'gestures': [{
            'id': g.id,
            'label': g.label,
            'speech': g.speech,
            'emoji': g.emoji,
            'hint': g.hint,
            'samples': int(counts[g.id]),
        } for g in gestures],
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    return path


def main():
    args = get_args()

    gestures = load_gestures()
    sync_label_file(gestures)
    num_classes = len(gestures)

    X, y = load_dataset(num_classes)
    print('Loaded {} samples across {} declared classes'.format(
        len(X), num_classes))

    counts = np.bincount(y, minlength=num_classes)
    empty = [g for g in gestures if counts[g.id] == 0]
    thin = [g for g in gestures if 0 < counts[g.id] < 30]

    for gesture in gestures:
        marker = ''
        if counts[gesture.id] == 0:
            marker = '  <-- NO DATA, cannot be recognised'
        elif counts[gesture.id] < 30:
            marker = '  <-- thin, collect more'
        print('  {:>2}  {:<12} {:>5}{}'.format(
            gesture.id, gesture.label, counts[gesture.id], marker))

    if empty:
        print('\nWARNING: {} of {} classes have no samples. Run '
              'collect_data.py for: {}'.format(
                  len(empty), num_classes,
                  ', '.join(g.label for g in empty)))
    if thin:
        print('WARNING: thin classes: {}'.format(
            ', '.join(g.label for g in thin)))

    X_train, X_test, y_train, y_test = stratified_split(X, y, args.test_size)
    print('\nTrain: {} samples   Test: {} samples'.format(
        len(X_train), len(X_test)))

    model = build_model(num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(HDF5_PATH, save_best_only=True,
                                           monitor='val_loss', verbose=0),
        tf.keras.callbacks.EarlyStopping(patience=40, monitor='val_loss',
                                         restore_best_weights=True, verbose=1),
    ]

    model.fit(X_train, y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=(X_test, y_test),
              callbacks=callbacks,
              verbose=2)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('\nHeld-out test accuracy: {:.1%}  (loss {:.4f})'.format(
        accuracy, loss))

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print_confusion(y_test, y_pred, gestures)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(TFLITE_PATH, 'wb') as f:
        f.write(converter.convert())

    web_path = export_web_weights(model, gestures, counts)

    print('\nWrote:')
    print('  {}'.format(HDF5_PATH))
    print('  {}'.format(TFLITE_PATH))
    print('  {}'.format(web_path))


if __name__ == '__main__':
    main()
