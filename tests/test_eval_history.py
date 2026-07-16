import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = (Path(__file__).resolve().parents[1] / 'paddleseg' / 'utils' /
               'eval_history.py')
SPEC = importlib.util.spec_from_file_location('eval_history', MODULE_PATH)
eval_history = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_history)


class EvalHistoryTest(unittest.TestCase):

    artifact_names = ('eval_metrics.json', 'miou_curve.png',
                      'defect_miss_over_curve.png')

    def load_json(self, output_dir):
        with (output_dir / 'eval_metrics.json').open(
                'r', encoding='utf-8') as history_file:
            return json.load(history_file)

    def assert_decodable_png(self, image_path):
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (760, 1280, 3))
        self.assertGreater(float(image.std()), 0.0)

    def assert_artifacts_absent(self, output_dir):
        for artifact_name in self.artifact_names:
            self.assertFalse((output_dir / artifact_name).exists())

    def test_persists_sorted_history_and_decodable_curves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / '\u8bad\u7ec3\u8f93\u51fa'
            eval_history.update_eval_history(
                output_dir,
                iteration=200,
                miou=0.72,
                miss_rate=0.18,
                over_rate=0.12,
                ema_miou=0.74,
                ema_miss_rate=0.16,
                ema_over_rate=0.10)
            eval_history.update_eval_history(
                output_dir,
                iteration=100,
                miou=0.61,
                miss_rate=0.30,
                over_rate=0.20)

            payload = self.load_json(output_dir)
            self.assertEqual(payload['version'], 1)
            self.assertEqual(
                [record['iteration'] for record in payload['records']],
                [100, 200])
            self.assertEqual(payload['records'][0], {
                'iteration': 100,
                'miou': 0.61,
                'miss_rate': 0.30,
                'over_rate': 0.20,
            })
            self.assertEqual(payload['records'][1]['ema_miou'], 0.74)
            self.assert_decodable_png(output_dir / 'miou_curve.png')
            self.assert_decodable_png(output_dir /
                                      'defect_miss_over_curve.png')

    def test_duplicate_iteration_fully_replaces_existing_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            eval_history.update_eval_history(
                output_dir,
                iteration=100,
                miou=0.50,
                miss_rate=0.40,
                over_rate=0.30,
                ema_miou=0.55,
                ema_miss_rate=0.35,
                ema_over_rate=0.25)
            eval_history.update_eval_history(
                output_dir,
                iteration=100,
                miou=0.80,
                miss_rate=0.10,
                over_rate=0.05)

            records = self.load_json(output_dir)['records']
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0], {
                'iteration': 100,
                'miou': 0.80,
                'miss_rate': 0.10,
                'over_rate': 0.05,
            })
            self.assert_decodable_png(output_dir / 'miou_curve.png')
            self.assert_decodable_png(output_dir /
                                      'defect_miss_over_curve.png')

    def test_new_training_removes_only_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            eval_history.update_eval_history(
                output_dir,
                iteration=100,
                miou=0.50,
                miss_rate=0.40,
                over_rate=0.30)
            unrelated_path = output_dir / 'model.pdparams'
            unrelated_path.write_bytes(b'model')

            result = eval_history.prepare_eval_history(output_dir)

            self.assertIsNone(result)
            self.assert_artifacts_absent(output_dir)
            self.assertEqual(unrelated_path.read_bytes(), b'model')

    def test_resume_truncates_history_and_regenerates_curves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for iteration, miou in ((100, 0.50), (200, 0.60), (300, 0.70)):
                eval_history.update_eval_history(
                    output_dir,
                    iteration=iteration,
                    miou=miou,
                    miss_rate=0.20,
                    over_rate=0.10)

            result = eval_history.prepare_eval_history(
                output_dir, resume_iteration=200)

            self.assertEqual(
                [record['iteration'] for record in result['records']],
                [100, 200])
            self.assertEqual(
                [record['iteration']
                 for record in self.load_json(output_dir)['records']],
                [100, 200])
            self.assert_decodable_png(output_dir / 'miou_curve.png')
            self.assert_decodable_png(output_dir /
                                      'defect_miss_over_curve.png')

    def test_resume_before_first_record_removes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            eval_history.update_eval_history(
                output_dir,
                iteration=100,
                miou=0.50,
                miss_rate=0.40,
                over_rate=0.30)

            result = eval_history.prepare_eval_history(
                output_dir, resume_iteration=50)

            self.assertIsNone(result)
            self.assert_artifacts_absent(output_dir)

    def test_resume_without_history_removes_orphan_curves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / 'miou_curve.png').write_bytes(b'stale')
            (output_dir / 'defect_miss_over_curve.png').write_bytes(b'stale')

            result = eval_history.prepare_eval_history(
                output_dir, resume_iteration=100)

            self.assertIsNone(result)
            self.assert_artifacts_absent(output_dir)

    def test_malformed_history_is_removed_and_can_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_dir.mkdir(exist_ok=True)
            (output_dir / 'eval_metrics.json').write_text(
                '{not valid json', encoding='utf-8')
            (output_dir / 'miou_curve.png').write_bytes(b'stale')
            (output_dir / 'defect_miss_over_curve.png').write_bytes(b'stale')

            result = eval_history.prepare_eval_history(
                output_dir, resume_iteration=100)

            self.assertIsNone(result)
            self.assert_artifacts_absent(output_dir)
            eval_history.update_eval_history(
                output_dir,
                iteration=100,
                miou=0.65,
                miss_rate=0.20,
                over_rate=0.10)
            self.assertEqual(
                self.load_json(output_dir)['records'][0]['iteration'], 100)
            self.assert_decodable_png(output_dir / 'miou_curve.png')
            self.assert_decodable_png(output_dir /
                                      'defect_miss_over_curve.png')


if __name__ == '__main__':
    unittest.main()
