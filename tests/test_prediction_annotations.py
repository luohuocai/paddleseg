import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image


MODULE_PATH = (Path(__file__).resolve().parents[1] / 'paddleseg' / 'utils' /
               'visualize.py')
SPEC = importlib.util.spec_from_file_location('seg_visualize', MODULE_PATH)
seg_visualize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seg_visualize)


class PredictionAnnotationsTest(unittest.TestCase):

    def test_palette_mask_keeps_class_ids_and_ignores_background(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_path = Path(temp_dir) / '\u6807\u7b7e.png'
            mask = np.asarray([[0, 1, 1, 255], [0, 2, 2, 255]],
                              dtype=np.uint8)
            palette_mask = Image.fromarray(mask, mode='P')
            palette = [0] * (256 * 3)
            palette[3:6] = [255, 0, 0]
            palette[6:9] = [0, 255, 0]
            palette_mask.putpalette(palette)
            palette_mask.save(mask_path)

            loaded = seg_visualize._load_class_id_mask(
                str(mask_path), mask.shape)
            np.testing.assert_array_equal(loaded, mask)
            objects = seg_visualize.segmentation_objects(
                loaded,
                class_names=['_background_', 'BL', 'GS'],
                min_area=1)
            self.assertEqual({item['class_id'] for item in objects}, {1, 2})
            self.assertEqual(
                {item['class_name'] for item in objects}, {'BL', 'GS'})

    def test_mask_resize_uses_nearest_class_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_path = Path(temp_dir) / 'small.png'
            mask = np.asarray([[0, 2], [2, 0]], dtype=np.uint8)
            Image.fromarray(mask, mode='L').save(mask_path)
            loaded = seg_visualize._load_class_id_mask(
                str(mask_path), (8, 8))
            self.assertEqual(set(np.unique(loaded).tolist()), {0, 2})

    def test_annotation_contains_distinct_gt_and_predicted_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / 'label.png'
            label = np.zeros((40, 60), dtype=np.uint8)
            label[4:20, 5:25] = 1
            Image.fromarray(label, mode='L').save(label_path)
            prediction = np.zeros_like(label)
            prediction[20:35, 30:55] = 2
            image = np.full((40, 60, 3), 100, dtype=np.uint8)

            original_put_text = cv2.putText
            with mock.patch.object(
                    seg_visualize.cv2,
                    'putText',
                    side_effect=original_put_text) as put_text:
                annotated = seg_visualize.annotate_segmentation_classes(
                    image,
                    prediction,
                    label_path=str(label_path),
                    class_names=['_background_', 'BL', 'GS'],
                    min_area=1)

            texts = [call.args[1] for call in put_text.call_args_list]
            self.assertIn('GT: BL', texts)
            self.assertIn('Pred: GS', texts)
            self.assertIn('GT:BL', texts)
            self.assertIn('P:GS', texts)
            self.assertEqual(annotated.shape[1], image.shape[1])
            self.assertGreater(annotated.shape[0], image.shape[0])

    def test_labelme_json_uses_shape_names_and_scales_coordinates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / 'sample.json'
            json_path.write_text(
                json.dumps({
                    'imageHeight': 100,
                    'imageWidth': 200,
                    'shapes': [{
                        'label': 'BL',
                        'shape_type': 'polygon',
                        'points': [[20, 10], [80, 10], [80, 50], [20, 50]],
                    }],
                }),
                encoding='utf-8')
            objects = seg_visualize._ground_truth_objects(
                str(json_path), (50, 100), ['_background_', 'BL'], 0, 255,
                1)
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0]['class_name'], 'BL')
            self.assertEqual(objects[0]['bbox'], (10, 5, 31, 21))

    def test_missing_ground_truth_is_shown_as_unavailable(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        prediction = np.zeros((20, 30), dtype=np.uint8)
        original_put_text = cv2.putText
        with mock.patch.object(
                seg_visualize.cv2,
                'putText',
                side_effect=original_put_text) as put_text:
            seg_visualize.annotate_segmentation_classes(
                image,
                prediction,
                class_names=['_background_'],
                min_area=1)
        texts = [call.args[1] for call in put_text.call_args_list]
        self.assertIn('GT: N/A', texts)
        self.assertIn('Pred: OK', texts)

    def test_defect_evaluation_visualization_matches_batch_rules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / 'label.png'
            label = np.zeros((100, 140), dtype=np.uint8)
            label[20:50, 10:40] = 1
            label[60:90, 80:110] = 1
            Image.fromarray(label, mode='L').save(label_path)

            prediction = np.zeros_like(label)
            prediction[22:48, 12:38] = 2
            prediction[60:80, 10:30] = 2
            image = np.full((100, 140, 3), 100, dtype=np.uint8)

            original_put_text = cv2.putText
            with mock.patch.object(
                    seg_visualize.cv2,
                    'putText',
                    side_effect=original_put_text) as put_text:
                rendered, stats = \
                    seg_visualize.annotate_defect_evaluation(
                        image,
                        prediction,
                        label_path=str(label_path),
                        class_names=['_background_', 'BL', 'GS'],
                        iou_threshold=0.1,
                        min_pred_area=1,
                        min_gt_area=1)

            self.assertEqual(stats['gt_object_count'], 2)
            self.assertEqual(stats['pred_object_count'], 2)
            self.assertEqual(stats['hit_object_count'], 1)
            self.assertEqual(stats['miss_object_count'], 1)
            self.assertEqual(stats['over_object_count'], 1)
            self.assertEqual(stats['hit_sample'], 1)
            self.assertEqual(stats['miss_sample'], 0)
            self.assertEqual(stats['over_sample'], 0)
            texts = [call.args[1] for call in put_text.call_args_list]
            self.assertIn('GT:BL', texts)
            self.assertIn('MISS GT:BL', texts)
            self.assertIn('P:GS', texts)
            self.assertIn(
                'GT=2 Pred=2 miss_sample=0 over_sample=0 miss_obj=1 over_obj=1',
                texts)
            self.assertEqual(rendered.shape, image.shape)

    def test_defect_evaluation_treats_unlabeled_images_as_ok(self):
        image = np.zeros((30, 40, 3), dtype=np.uint8)
        prediction = np.zeros((30, 40), dtype=np.uint8)
        _, stats = seg_visualize.annotate_defect_evaluation(
            image,
            prediction,
            class_names=['_background_'],
            min_pred_area=1,
            min_gt_area=1)
        self.assertEqual(stats['hit_sample'], 1)
        self.assertEqual(stats['miss_sample'], 0)
        self.assertEqual(stats['over_sample'], 0)

        prediction[5:15, 5:15] = 1
        _, stats = seg_visualize.annotate_defect_evaluation(
            image,
            prediction,
            class_names=['_background_', 'defect'],
            min_pred_area=1,
            min_gt_area=1)
        self.assertEqual(stats['hit_sample'], 0)
        self.assertEqual(stats['miss_sample'], 0)
        self.assertEqual(stats['over_sample'], 1)
        self.assertEqual(stats['over_object_count'], 1)

    def test_defect_evaluation_reads_unicode_image_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / '测试图片.png'
            image = np.full((24, 32, 3), 80, dtype=np.uint8)
            encoded, buffer = cv2.imencode('.png', image)
            self.assertTrue(encoded)
            buffer.tofile(str(image_path))
            prediction = np.zeros((24, 32), dtype=np.uint8)
            rendered, stats = seg_visualize.annotate_defect_evaluation(
                str(image_path),
                prediction,
                class_names=['_background_'],
                min_pred_area=1,
                min_gt_area=1)
            self.assertEqual(rendered.shape, image.shape)
            self.assertEqual(stats['hit_sample'], 1)


if __name__ == '__main__':
    unittest.main()
