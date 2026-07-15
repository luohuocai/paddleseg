import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (Path(__file__).resolve().parents[1] / 'paddleseg' / 'utils' /
               'defect_metrics.py')
SPEC = importlib.util.spec_from_file_location('defect_metrics', MODULE_PATH)
defect_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(defect_metrics)


class DefectMetricsTest(unittest.TestCase):

    def evaluate(self, pred, label, threshold=0.1, **kwargs):
        return defect_metrics.evaluate_sample(
            np.asarray(pred, dtype=np.uint8),
            np.asarray(label, dtype=np.uint8),
            hit_iou_threshold=threshold,
            **kwargs)

    def test_ok_sample_without_prediction_is_hit(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        stats = self.evaluate(mask, mask)
        self.assertEqual(stats['hit_sample'], 1)
        self.assertEqual(stats['miss_sample'], 0)
        self.assertEqual(stats['over_sample'], 0)

    def test_ok_sample_with_prediction_is_over_detected(self):
        pred = np.zeros((8, 8), dtype=np.uint8)
        pred[1:4, 1:4] = 1
        stats = self.evaluate(pred, np.zeros_like(pred))
        self.assertEqual(stats['over_sample'], 1)
        self.assertEqual(stats['over_object_total'], 1)

    def test_positive_sample_without_prediction_is_missed(self):
        label = np.zeros((8, 8), dtype=np.uint8)
        label[1:4, 1:4] = 1
        stats = self.evaluate(np.zeros_like(label), label)
        self.assertEqual(stats['miss_sample'], 1)
        self.assertEqual(stats['miss_object_total'], 1)
        self.assertEqual(stats['over_sample'], 0)

    def test_overlapping_prediction_hits_positive_sample(self):
        pred = np.zeros((8, 8), dtype=np.uint8)
        label = np.zeros((8, 8), dtype=np.uint8)
        pred[2:5, 2:5] = 1
        label[1:5, 1:5] = 1
        stats = self.evaluate(pred, label)
        self.assertEqual(stats['hit_sample'], 1)
        self.assertEqual(stats['hit_object_total'], 1)
        self.assertEqual(stats['miss_sample'], 0)
        self.assertEqual(stats['over_sample'], 0)

    def test_partial_object_hit_is_not_a_missed_sample(self):
        pred = np.zeros((12, 12), dtype=np.uint8)
        label = np.zeros((12, 12), dtype=np.uint8)
        pred[1:4, 1:4] = 1
        label[1:4, 1:4] = 1
        label[8:11, 8:11] = 1
        stats = self.evaluate(pred, label)
        self.assertEqual(stats['hit_sample'], 1)
        self.assertEqual(stats['miss_sample'], 0)
        self.assertEqual(stats['hit_object_total'], 1)
        self.assertEqual(stats['miss_object_total'], 1)

    def test_unmatched_positive_prediction_is_both_miss_and_over(self):
        pred = np.zeros((12, 12), dtype=np.uint8)
        label = np.zeros((12, 12), dtype=np.uint8)
        pred[1:4, 1:4] = 1
        label[8:11, 8:11] = 1
        stats = self.evaluate(pred, label)
        self.assertEqual(stats['miss_sample'], 1)
        self.assertEqual(stats['over_sample'], 1)
        self.assertEqual(stats['miss_object_total'], 1)
        self.assertEqual(stats['over_object_total'], 1)

    def test_class_name_is_ignored_for_matching(self):
        pred = np.zeros((8, 8), dtype=np.uint8)
        label = np.zeros((8, 8), dtype=np.uint8)
        pred[1:5, 1:5] = 2
        label[1:5, 1:5] = 1
        stats = self.evaluate(pred, label)
        self.assertEqual(stats['hit_object_total'], 1)
        self.assertEqual(stats['hit_sample'], 1)

    def test_prediction_in_ignored_region_is_not_over_detected(self):
        pred = np.zeros((8, 8), dtype=np.uint8)
        label = np.zeros((8, 8), dtype=np.uint8)
        pred[1:5, 1:5] = 1
        label[1:5, 1:5] = 255
        stats = self.evaluate(pred, label)
        self.assertEqual(stats['pred_object_total'], 0)
        self.assertEqual(stats['over_sample'], 0)
        self.assertEqual(stats['hit_sample'], 1)

    def test_minimum_prediction_area_filters_small_components(self):
        pred = np.zeros((8, 8), dtype=np.uint8)
        pred[2, 2] = 1
        stats = self.evaluate(
            pred, np.zeros_like(pred), min_pred_area=2)
        self.assertEqual(stats['pred_object_total'], 0)
        self.assertEqual(stats['over_sample'], 0)


if __name__ == '__main__':
    unittest.main()
