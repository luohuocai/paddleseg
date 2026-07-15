# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import cv2
import numpy as np


STAT_KEYS = (
    'sample_total',
    'positive_sample_total',
    'ok_sample_total',
    'gt_object_total',
    'pred_object_total',
    'hit_object_total',
    'miss_object_total',
    'over_object_total',
    'hit_sample',
    'miss_sample',
    'over_sample',
)


def empty_stats():
    return {key: 0 for key in STAT_KEYS}


def merge_stats(total, current):
    for key in STAT_KEYS:
        total[key] += current[key]


def stats_to_array(stats):
    return np.asarray([stats[key] for key in STAT_KEYS], dtype=np.int64)


def stats_from_array(values):
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if len(values) != len(STAT_KEYS):
        raise ValueError('Unexpected defect statistics length: {}.'.format(
            len(values)))
    return {key: int(value) for key, value in zip(STAT_KEYS, values)}


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _contour_iou(a, b):
    contour_a, bbox_a = a
    contour_b, bbox_b = b
    if contour_a is None or contour_b is None:
        return _bbox_iou(bbox_a, bbox_b)
    if _bbox_iou(bbox_a, bbox_b) == 0:
        return 0.0

    x1 = int(math.floor(min(bbox_a[0], bbox_b[0])))
    y1 = int(math.floor(min(bbox_a[1], bbox_b[1])))
    x2 = int(math.ceil(max(bbox_a[2], bbox_b[2])))
    y2 = int(math.ceil(max(bbox_a[3], bbox_b[3])))
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return 0.0

    offset = np.asarray([[x1, y1]], dtype=np.float32)
    contour_a = np.rint(contour_a - offset).astype(np.int32)
    contour_b = np.rint(contour_b - offset).astype(np.int32)
    mask_a = np.zeros((height, width), dtype=np.uint8)
    mask_b = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask_a, [contour_a.reshape(-1, 1, 2)], 1)
    cv2.fillPoly(mask_b, [contour_b.reshape(-1, 1, 2)], 1)

    intersection = np.count_nonzero((mask_a > 0) & (mask_b > 0))
    if intersection == 0:
        return 0.0
    union = np.count_nonzero((mask_a > 0) | (mask_b > 0))
    return float(intersection / union) if union > 0 else 0.0


def _component_objects(binary_mask, min_area):
    objects = []
    count, component_map, component_stats, _ = \
        cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    for component_id in range(1, count):
        area = int(component_stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(component_stats[component_id, cv2.CC_STAT_LEFT])
        y = int(component_stats[component_id, cv2.CC_STAT_TOP])
        width = int(component_stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(component_stats[component_id, cv2.CC_STAT_HEIGHT])
        component_roi = (component_map[y:y + height, x:x + width] ==
                         component_id).astype(np.uint8)
        contours, _ = cv2.findContours(component_roi, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contour = None
        if contours:
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
            contour = contour.astype(np.float32)
            contour[:, 0] += float(x)
            contour[:, 1] += float(y)
            if contour.shape[0] < 3 or abs(cv2.contourArea(contour)) <= 0:
                contour = None

        bbox = (float(x), float(y), float(x + width), float(y + height))
        objects.append((contour, bbox))
    return objects


def _mask_to_objects(mask, min_area, ignore_index, use_multilabel):
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    objects = []
    if use_multilabel:
        channels = mask if mask.ndim == 3 else mask[np.newaxis, ...]
        for channel in channels:
            binary_mask = (channel > 0).astype(np.uint8)
            objects.extend(_component_objects(binary_mask, min_area))
        return objects

    if mask.ndim != 2:
        raise ValueError('Defect evaluation expects a 2-D class mask, but '
                         'received shape {}.'.format(mask.shape))

    for class_id in np.unique(mask):
        if class_id == 0 or class_id == ignore_index:
            continue
        binary_mask = (mask == class_id).astype(np.uint8)
        objects.extend(_component_objects(binary_mask, min_area))
    return objects


def _evaluate_matches(pred_objects, gt_objects, hit_iou_threshold):
    gt_hit = [False] * len(gt_objects)
    pred_hit = [False] * len(pred_objects)

    for gt_index, gt_object in enumerate(gt_objects):
        for pred_index, pred_object in enumerate(pred_objects):
            if _contour_iou(gt_object, pred_object) >= hit_iou_threshold:
                gt_hit[gt_index] = True
                pred_hit[pred_index] = True
    return gt_hit, pred_hit


def evaluate_sample(pred,
                    label,
                    hit_iou_threshold=0.1,
                    min_pred_area=1,
                    min_gt_area=1,
                    ignore_index=255,
                    use_multilabel=False):
    if not 0 <= hit_iou_threshold <= 1:
        raise ValueError('hit_iou_threshold must be between 0 and 1.')
    if min_pred_area < 1 or min_gt_area < 1:
        raise ValueError('Minimum component areas must be positive integers.')

    if not use_multilabel:
        pred = np.asarray(pred)
        label = np.asarray(label)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if label.ndim == 3 and label.shape[0] == 1:
            label = label[0]
        if pred.shape == label.shape:
            pred = pred.copy()
            pred[label == ignore_index] = 0

    pred_objects = _mask_to_objects(pred, min_pred_area, ignore_index,
                                    use_multilabel)
    gt_objects = _mask_to_objects(label, min_gt_area, ignore_index,
                                  use_multilabel)
    gt_hit, pred_hit = _evaluate_matches(pred_objects, gt_objects,
                                         hit_iou_threshold)

    hit_objects = sum(gt_hit)
    miss_objects = sum(not is_hit for is_hit in gt_hit)
    over_objects = sum(not is_hit for is_hit in pred_hit)
    has_gt = len(gt_objects) > 0
    has_hit_gt = hit_objects > 0
    miss_sample = int(has_gt and not has_hit_gt)
    over_sample = int((not has_gt and len(pred_objects) > 0) or
                      (has_gt and not has_hit_gt and over_objects > 0))
    hit_sample = int(has_hit_gt or
                     (not has_gt and len(pred_objects) == 0))

    stats = empty_stats()
    stats.update({
        'sample_total': 1,
        'positive_sample_total': int(has_gt),
        'ok_sample_total': int(not has_gt),
        'gt_object_total': len(gt_objects),
        'pred_object_total': len(pred_objects),
        'hit_object_total': hit_objects,
        'miss_object_total': miss_objects,
        'over_object_total': over_objects,
        'hit_sample': hit_sample,
        'miss_sample': miss_sample,
        'over_sample': over_sample,
    })
    return stats


def sample_rates(stats):
    total = stats['sample_total']
    if total == 0:
        return {'hit_rate': 0.0, 'miss_rate': 0.0, 'over_rate': 0.0}
    return {
        'hit_rate': stats['hit_sample'] / total,
        'miss_rate': stats['miss_sample'] / total,
        'over_rate': stats['over_sample'] / total,
    }
