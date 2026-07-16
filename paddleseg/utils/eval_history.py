# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persist validation metrics and render training-time evaluation curves."""

import json
import math
import os

import cv2
import numpy as np


_HISTORY_FILE = 'eval_metrics.json'
_MIOU_CURVE_FILE = 'miou_curve.png'
_DEFECT_CURVE_FILE = 'defect_miss_over_curve.png'
_OPTIONAL_EMA_KEYS = ('ema_miou', 'ema_miss_rate', 'ema_over_rate')

_CANVAS_WIDTH = 1280
_CANVAS_HEIGHT = 760
_PLOT_LEFT = 105
_PLOT_RIGHT = 1235
_PLOT_TOP = 145
_PLOT_BOTTOM = 650


def _as_fraction(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('{} must be a number.'.format(name))
    if not math.isfinite(value):
        raise ValueError('{} must be finite.'.format(name))
    if value < 0.0 or value > 1.0:
        raise ValueError('{} must be between 0 and 1.'.format(name))
    return value


def _normalize_record(record):
    if not isinstance(record, dict):
        raise ValueError('Each evaluation history record must be an object.')

    try:
        iteration = int(record['iteration'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('Each evaluation history record needs an iteration.')
    if iteration < 0:
        raise ValueError('iteration must be non-negative.')

    try:
        normalized = {
            'iteration': iteration,
            'miou': _as_fraction(record['miou'], 'miou'),
            'miss_rate': _as_fraction(record['miss_rate'], 'miss_rate'),
            'over_rate': _as_fraction(record['over_rate'], 'over_rate'),
        }
    except KeyError as error:
        raise ValueError('Evaluation history record is missing {}.'.format(
            error.args[0]))
    for key in _OPTIONAL_EMA_KEYS:
        if key in record and record[key] is not None:
            normalized[key] = _as_fraction(record[key], key)
    return normalized


def _load_records(history_path):
    if not os.path.exists(history_path):
        return []

    with open(history_path, 'r', encoding='utf-8') as history_file:
        payload = json.load(history_file)
    if not isinstance(payload, dict) or not isinstance(
            payload.get('records'), list):
        raise ValueError(
            'Evaluation history must contain a JSON records list.')

    records_by_iteration = {}
    for raw_record in payload['records']:
        record = _normalize_record(raw_record)
        records_by_iteration[record['iteration']] = record
    return [
        records_by_iteration[key] for key in sorted(records_by_iteration)
    ]


def _atomic_write_json(path, payload):
    temporary_path = path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8', newline='\n') as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write('\n')
    os.replace(temporary_path, path)


def _atomic_write_png(path, image):
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise RuntimeError('Failed to encode evaluation curve: {}.'.format(
            path))
    temporary_path = path + '.tmp'
    encoded.tofile(temporary_path)
    os.replace(temporary_path, path)


def _text_width(text, scale=0.55, thickness=1):
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                              thickness)
    return size[0]


def _draw_centered_text(image,
                        text,
                        center_x,
                        baseline_y,
                        scale=0.55,
                        color=(45, 45, 45),
                        thickness=1):
    x = int(center_x - _text_width(text, scale, thickness) / 2)
    cv2.putText(image, text, (x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _tick_indices(record_count, max_ticks=8):
    if record_count <= max_ticks:
        return list(range(record_count))
    indices = []
    for tick_index in range(max_ticks):
        index = int(round(tick_index * (record_count - 1) /
                          float(max_ticks - 1)))
        if not indices or index != indices[-1]:
            indices.append(index)
    return indices


def _x_coordinate(iteration, minimum_iteration, maximum_iteration):
    if minimum_iteration == maximum_iteration:
        return int((_PLOT_LEFT + _PLOT_RIGHT) / 2)
    fraction = ((iteration - minimum_iteration) /
                float(maximum_iteration - minimum_iteration))
    return int(round(_PLOT_LEFT + fraction * (_PLOT_RIGHT - _PLOT_LEFT)))


def _y_coordinate(fraction):
    percentage = max(0.0, min(100.0, float(fraction) * 100.0))
    return int(round(_PLOT_BOTTOM - percentage / 100.0 *
                     (_PLOT_BOTTOM - _PLOT_TOP)))


def _draw_legend(image, series):
    x = _PLOT_LEFT
    y = 82
    for label, _, color in series:
        item_width = 62 + _text_width(label, 0.58, 1)
        if x + item_width > _PLOT_RIGHT:
            x = _PLOT_LEFT
            y += 34
        cv2.line(image, (x, y - 5), (x + 34, y - 5), color, 3,
                 cv2.LINE_AA)
        cv2.circle(image, (x + 17, y - 5), 5, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (x + 44, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (45, 45, 45), 1,
                    cv2.LINE_AA)
        x += item_width + 24


def _draw_axes(image, records, title, y_label):
    cv2.putText(image, title, (_PLOT_LEFT, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.92, (25, 25, 25), 2,
                cv2.LINE_AA)
    cv2.putText(image, y_label, (_PLOT_LEFT, _PLOT_TOP - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (45, 45, 45), 1,
                cv2.LINE_AA)

    for percentage in range(0, 101, 20):
        y = _y_coordinate(percentage / 100.0)
        cv2.line(image, (_PLOT_LEFT, y), (_PLOT_RIGHT, y),
                 (224, 224, 224), 1, cv2.LINE_AA)
        cv2.putText(image, str(percentage), (_PLOT_LEFT - 52, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1,
                    cv2.LINE_AA)

    cv2.rectangle(image, (_PLOT_LEFT, _PLOT_TOP),
                  (_PLOT_RIGHT, _PLOT_BOTTOM), (65, 65, 65), 2,
                  cv2.LINE_AA)

    minimum_iteration = records[0]['iteration']
    maximum_iteration = records[-1]['iteration']
    for record_index in _tick_indices(len(records)):
        iteration = records[record_index]['iteration']
        x = _x_coordinate(iteration, minimum_iteration, maximum_iteration)
        cv2.line(image, (x, _PLOT_BOTTOM), (x, _PLOT_BOTTOM + 8),
                 (65, 65, 65), 1, cv2.LINE_AA)
        _draw_centered_text(image, str(iteration), x, _PLOT_BOTTOM + 34,
                            scale=0.5)
    _draw_centered_text(image, 'Iteration',
                        int((_PLOT_LEFT + _PLOT_RIGHT) / 2),
                        _PLOT_BOTTOM + 76,
                        scale=0.62,
                        thickness=1)


def _draw_series(image, records, key, color):
    minimum_iteration = records[0]['iteration']
    maximum_iteration = records[-1]['iteration']
    previous_point = None
    for record in records:
        value = record.get(key)
        if value is None:
            previous_point = None
            continue
        point = (_x_coordinate(record['iteration'], minimum_iteration,
                               maximum_iteration), _y_coordinate(value))
        if previous_point is not None:
            cv2.line(image, previous_point, point, color, 3, cv2.LINE_AA)
        cv2.circle(image, point, 5, color, -1, cv2.LINE_AA)
        cv2.circle(image, point, 7, color, 1, cv2.LINE_AA)
        previous_point = point


def _render_curve(records, title, y_label, series):
    image = np.full((_CANVAS_HEIGHT, _CANVAS_WIDTH, 3), 255,
                    dtype=np.uint8)
    _draw_axes(image, records, title, y_label)
    _draw_legend(image, series)
    for _, key, color in series:
        _draw_series(image, records, key, color)
    return image


def _artifact_paths(save_dir):
    return (os.path.join(save_dir, _HISTORY_FILE),
            os.path.join(save_dir, _MIOU_CURVE_FILE),
            os.path.join(save_dir, _DEFECT_CURVE_FILE))


def _remove_artifacts(save_dir):
    for artifact_path in _artifact_paths(save_dir):
        try:
            os.remove(artifact_path)
        except FileNotFoundError:
            pass


def _persist_records(save_dir, records):
    payload = {'version': 1, 'records': records}
    history_path, miou_path, defect_path = _artifact_paths(save_dir)
    _atomic_write_json(history_path, payload)

    miou_series = [('mIoU', 'miou', (210, 105, 30))]
    if any('ema_miou' in record for record in records):
        miou_series.append(('EMA mIoU', 'ema_miou', (45, 155, 45)))
    miou_curve = _render_curve(records, 'Validation mIoU', 'mIoU (%)',
                               miou_series)
    _atomic_write_png(miou_path, miou_curve)

    defect_series = [('Miss', 'miss_rate', (55, 55, 220)),
                     ('Over', 'over_rate', (0, 145, 255))]
    if any('ema_miss_rate' in record for record in records):
        defect_series.append(
            ('EMA Miss', 'ema_miss_rate', (185, 80, 185)))
    if any('ema_over_rate' in record for record in records):
        defect_series.append(
            ('EMA Over', 'ema_over_rate', (175, 165, 25)))
    defect_curve = _render_curve(records,
                                 'Sample-level Miss / Over Detection Rates',
                                 'Rate (%)', defect_series)
    _atomic_write_png(defect_path, defect_curve)
    return payload


def prepare_eval_history(save_dir, resume_iteration=None):
    """Prepare evaluation artifacts before a new or resumed training run.

    A new run (``resume_iteration=None``) removes stale history and curves. A
    resumed run retains records up to and including ``resume_iteration`` and
    regenerates both curves. Missing, empty, or malformed history is treated
    as an empty history and all three artifacts are removed.

    Returns:
        dict|None: The truncated payload, or ``None`` for an empty history.
    """
    save_dir = os.fspath(save_dir)
    if resume_iteration is None:
        _remove_artifacts(save_dir)
        return None

    if isinstance(resume_iteration, bool):
        raise ValueError('resume_iteration must be a non-negative integer.')
    try:
        normalized_iteration = int(resume_iteration)
    except (TypeError, ValueError):
        raise ValueError('resume_iteration must be a non-negative integer.')
    if normalized_iteration < 0 or normalized_iteration != resume_iteration:
        raise ValueError('resume_iteration must be a non-negative integer.')

    history_path = os.path.join(save_dir, _HISTORY_FILE)
    try:
        records = _load_records(history_path)
    except ValueError:
        _remove_artifacts(save_dir)
        return None

    records = [
        record for record in records
        if record['iteration'] <= normalized_iteration
    ]
    if not records:
        _remove_artifacts(save_dir)
        return None

    os.makedirs(save_dir, exist_ok=True)
    return _persist_records(save_dir, records)


def update_eval_history(save_dir,
                        iteration,
                        miou,
                        miss_rate,
                        over_rate,
                        ema_miou=None,
                        ema_miss_rate=None,
                        ema_over_rate=None):
    """Upsert one evaluation and regenerate JSON and PNG curve artifacts.

    Metric values are fractions in the inclusive range ``[0, 1]``. Curves
    display those values as percentages. Existing records are loaded on every
    call so a resumed training process retains earlier evaluation points.

    Returns:
        dict: The persisted payload containing records sorted by iteration.
    """
    save_dir = os.fspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    history_path = os.path.join(save_dir, _HISTORY_FILE)
    records = _load_records(history_path)

    new_record = _normalize_record({
        'iteration': iteration,
        'miou': miou,
        'miss_rate': miss_rate,
        'over_rate': over_rate,
        'ema_miou': ema_miou,
        'ema_miss_rate': ema_miss_rate,
        'ema_over_rate': ema_over_rate,
    })
    records_by_iteration = {
        record['iteration']: record
        for record in records
    }
    records_by_iteration[new_record['iteration']] = new_record
    records = [
        records_by_iteration[key] for key in sorted(records_by_iteration)
    ]

    return _persist_records(save_dir, records)
