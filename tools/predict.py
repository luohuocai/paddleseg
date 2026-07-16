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

import argparse
import os

import paddle

from paddleseg.core import predict
from paddleseg.cvlibs import Config, SegBuilder, manager
from paddleseg.transforms import Compose
from paddleseg.utils import (get_image_list_with_labels, get_sys_env, logger,
                             utils)


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of trained weights for prediction.',
        type=str)
    parser.add_argument(
        '--image_path',
        help='The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the predicted results.',
        type=str,
        default='./output/result')
    parser.add_argument(
        '--device',
        help='Set the device place for predicting model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)
    parser.add_argument(
        '--device_id',
        help='Set the device id for predicting model.',
        default=0,
        type=int)
    parser.add_argument(
        '--annotate_classes',
        action='store_true',
        help=(
            'Annotate added_prediction images with ground-truth and predicted '
            'class names. Ground truth is read from the second column of a '
            'PaddleSeg file list.'))
    parser.add_argument(
        '--class_names',
        type=str,
        help=(
            'Optional class_names.txt whose line index is the class id. When '
            'omitted, dataset-local class_names.txt locations are searched.'))
    parser.add_argument(
        '--label_dir',
        type=str,
        help=(
            'Optional directory containing ground-truth class-id masks or '
            'LabelMe JSON files. Use this when image and label directories '
            'are separate, for example images/val and labels/val.'))
    parser.add_argument(
        '--annotation_min_area',
        type=int,
        default=20,
        help='Minimum connected-component area to annotate. Default: 20.')
    parser.add_argument(
        '--background_id',
        type=int,
        default=0,
        help='Class id treated as background when annotating. Default: 0.')
    parser.add_argument(
        '--ignore_index',
        type=int,
        default=255,
        help='Ground-truth class id ignored when annotating. Default: 255.')
    parser.add_argument(
        '--defect_eval_visualization',
        action='store_true',
        help=(
            'Save GT/prediction hit, miss, and over-detection visualizations. '
            'This is also enabled by test_config.defect_eval.'))
    parser.add_argument(
        '--defect_iou_threshold',
        type=float,
        default=None,
        help=(
            'Contour IoU threshold used by defect visualization. It overrides '
            'test_config.defect_iou_threshold when provided.'))
    parser.add_argument(
        '--defect_min_pred_area',
        type=int,
        default=None,
        help='Minimum predicted component area used by defect visualization.')
    parser.add_argument(
        '--defect_min_gt_area',
        type=int,
        default=None,
        help='Minimum GT component area used by defect visualization.')

    # Data augment params
    parser.add_argument(
        '--aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        nargs='+',
        help='Scales for augment, e.g., `--scales 0.75 1.0 1.25`.',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # Sliding window evaluation params
    parser.add_argument(
        '--is_slide',
        help='Whether to predict images in sliding window method',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        nargs=2,
        help='The crop size of sliding window, the first is width and the second is height.'
        'For example, `--crop_size 512 512`',
        type=int)
    parser.add_argument(
        '--stride',
        nargs=2,
        help='The stride of sliding window, the first is width and the second is height.'
        'For example, `--stride 512 512`',
        type=int)

    # Custom color map
    parser.add_argument(
        '--custom_color',
        nargs='+',
        help='Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int)

    # Set multi-label mode
    parser.add_argument(
        '--use_multilabel',
        action='store_true',
        default=False,
        help='Whether to enable multilabel mode. Default: False.')

    return parser.parse_args()


def merge_test_config(cfg, args):
    test_config = cfg.test_config
    if 'aug_eval' in test_config:
        test_config.pop('aug_eval')
    if 'auc_roc' in test_config:
        test_config.pop('auc_roc')
    supported_defect_options = {
        'defect_eval', 'defect_iou_threshold', 'defect_min_pred_area',
        'defect_min_gt_area'
    }
    for key in list(test_config.keys()):
        if key.startswith('defect_') and key not in supported_defect_options:
            test_config.pop(key)
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical
    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride
    if args.custom_color:
        test_config['custom_color'] = args.custom_color
    if args.use_multilabel:
        test_config['use_multilabel'] = args.use_multilabel
    if args.defect_eval_visualization:
        test_config['defect_eval'] = True
    if args.defect_iou_threshold is not None:
        test_config['defect_iou_threshold'] = args.defect_iou_threshold
    if args.defect_min_pred_area is not None:
        test_config['defect_min_pred_area'] = args.defect_min_pred_area
    if args.defect_min_gt_area is not None:
        test_config['defect_min_gt_area'] = args.defect_min_gt_area
    return test_config


def load_class_names(class_names_path,
                     image_path,
                     num_classes,
                     label_dir=None):
    """Load class-id names, with a dataset-local automatic fallback."""
    resolved_path = class_names_path
    if resolved_path is None:
        if os.path.isdir(image_path):
            dataset_dir = os.path.abspath(image_path)
        else:
            dataset_dir = os.path.dirname(os.path.abspath(image_path))
        candidates = []
        if label_dir is not None:
            try:
                common_dir = os.path.commonpath(
                    [dataset_dir, os.path.abspath(label_dir)])
            except ValueError:
                common_dir = None
            candidates.append(os.path.join(label_dir, 'class_names.txt'))
            candidates.append(
                os.path.join(os.path.dirname(os.path.abspath(label_dir)),
                             'class_names.txt'))
            if common_dir is not None:
                candidates.append(os.path.join(common_dir,
                                               'class_names.txt'))
        candidates.extend([
            os.path.join(dataset_dir, 'labels', 'class_names.txt'),
            os.path.join(dataset_dir, 'class_names.txt'),
        ])
        for candidate in candidates:
            if os.path.isfile(candidate):
                resolved_path = candidate
                break

    if resolved_path is None:
        logger.warning(
            'No class_names.txt was found. Annotation will use class_<id>.')
        return ['class_{}'.format(index) for index in range(num_classes)]

    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(
            'Class names file does not exist: {}'.format(resolved_path))
    with open(resolved_path, 'r', encoding='utf-8-sig') as names_file:
        class_names = [line.strip() for line in names_file.read().splitlines()]

    original_count = len(class_names)
    if num_classes and original_count != num_classes:
        logger.warning(
            'class_names.txt contains {} lines, but the model has {} classes. '
            'Each line must use the same class-id order as the training masks.'.
            format(original_count, num_classes))
    if len(class_names) < num_classes:
        class_names.extend([
            'class_{}'.format(index)
            for index in range(len(class_names), num_classes)
        ])
    logger.info('Use class names from {}'.format(resolved_path))
    logger.info('Class mapping: {}'.format(', '.join(
        '{}={}'.format(index, name or 'class_{}'.format(index))
        for index, name in enumerate(class_names))))
    return class_names


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config)
    builder = SegBuilder(cfg)
    test_config = merge_test_config(cfg, args)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    if args.device != 'cpu':
        device = f"{args.device}:{args.device_id}"
    else:
        device = args.device
    utils.set_device(device)

    model = builder.model
    transforms = Compose(builder.val_transforms)
    if args.label_dir is not None and not os.path.isdir(args.label_dir):
        raise FileNotFoundError(
            'Ground-truth label directory does not exist: {}'.format(
                args.label_dir))
    image_list, image_dir, label_map = get_image_list_with_labels(
        args.image_path, label_dir=args.label_dir)
    logger.info('The number of images: {}'.format(len(image_list)))

    defect_eval = bool(test_config.get('defect_eval', False))
    class_names = None
    if args.annotate_classes or defect_eval:
        if args.annotation_min_area < 1:
            raise ValueError('--annotation_min_area must be at least 1.')
        defect_iou_threshold = test_config.get('defect_iou_threshold', 0.1)
        defect_min_pred_area = test_config.get('defect_min_pred_area', 1)
        defect_min_gt_area = test_config.get('defect_min_gt_area', 1)
        if not 0 <= defect_iou_threshold <= 1:
            raise ValueError('--defect_iou_threshold must be between 0 and 1.')
        if defect_min_pred_area < 1 or defect_min_gt_area < 1:
            raise ValueError('Defect minimum component areas must be at least 1.')
        model_cfg = cfg.dic.get('model', {})
        val_dataset_cfg = cfg.dic.get('val_dataset', {})
        num_classes = model_cfg.get('num_classes',
                                    val_dataset_cfg.get('num_classes', 0))
        class_names = load_class_names(args.class_names, args.image_path,
                                       num_classes, args.label_dir)
        if args.annotate_classes and not label_map:
            logger.warning(
                'No ground-truth paths were found in --image_path. Predicted '
                'classes will be annotated, and GT will be shown as N/A.')
        missing_label_count = len(image_list) - len(label_map)
        if defect_eval and missing_label_count:
            logger.warning(
                '{} of {} inputs have no ground-truth path. Defect '
                'visualization will treat those inputs as unlabeled OK '
                'samples, matching the standalone batch evaluator.'.format(
                    missing_label_count, len(image_list)))

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        label_map=label_map,
        class_names=class_names,
        annotate_classes=args.annotate_classes,
        annotation_min_area=args.annotation_min_area,
        background_id=args.background_id,
        ignore_index=args.ignore_index,
        **test_config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
