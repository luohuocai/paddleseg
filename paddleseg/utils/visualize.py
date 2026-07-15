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

import json
import os

import cv2
import numpy as np
from PIL import Image as PILImage


def visualize(image, result, color_map, save_dir=None, weight=0.6, use_multilabel=False):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")

    im = cv2.imread(image)
    if not use_multilabel:
        # Use OpenCV LUT for color mapping
        c1 = cv2.LUT(result, color_map[:, 0])
        c2 = cv2.LUT(result, color_map[:, 1])
        c3 = cv2.LUT(result, color_map[:, 2])
        pseudo_img = np.dstack((c3, c2, c1))

        vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)
    else:
        vis_result = im.copy()
        for i in range(result.shape[0]):
            mask = result[i]
            c1 = np.where(mask, color_map[i, 0], vis_result[..., 0])
            c2 = np.where(mask, color_map[i, 1], vis_result[..., 1])
            c3 = np.where(mask, color_map[i, 2], vis_result[..., 2])
            pseudo_img = np.dstack((c3, c2, c1)).astype('uint8')

            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vis_result = cv2.addWeighted(vis_result, weight, pseudo_img, 1 - weight, 0)
            contour_color = (int(color_map[i, 0]), int(color_map[i, 1]), int(color_map[i, 2]))
            vis_result = cv2.drawContours(vis_result, contour, -1, contour_color, 1)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result


def _class_name_from_id(class_names, class_id):
    if class_names is not None and 0 <= class_id < len(class_names):
        class_name = str(class_names[class_id]).strip()
        if class_name:
            return class_name
    return 'class_{}'.format(class_id)


def _component_objects(binary_mask, class_id, class_name, min_area):
    """Extract connected components used by annotated prediction images."""
    binary_mask = np.asarray(binary_mask, dtype=np.uint8)
    count, component_map, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8)
    objects = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component = (component_map[y:y + height, x:x + width] ==
                     component_id).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour[:, :, 0] += x
            contour[:, :, 1] += y
        else:
            contour = np.asarray([[[x, y]], [[x + width, y]],
                                  [[x + width, y + height]],
                                  [[x, y + height]]],
                                 dtype=np.int32)
        objects.append({
            'class_id': int(class_id),
            'class_name': class_name,
            'area': area,
            'bbox': (x, y, width, height),
            'contour': contour,
        })
    return objects


def segmentation_objects(mask,
                         class_names=None,
                         background_id=0,
                         ignore_index=255,
                         min_area=1,
                         use_multilabel=False):
    """Convert a segmentation mask into labeled connected components."""
    mask = np.asarray(mask)
    objects = []
    if use_multilabel and mask.ndim == 3:
        for channel_id, channel in enumerate(mask):
            class_id = channel_id + 1
            class_name = _class_name_from_id(class_names, class_id)
            objects.extend(
                _component_objects(channel > 0, class_id, class_name,
                                   min_area))
        return objects

    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(
            'Class annotation expects a 2-D class-id mask, but got shape {}.'.
            format(mask.shape))
    for value in np.unique(mask).tolist():
        class_id = int(value)
        if class_id in (background_id, ignore_index):
            continue
        class_name = _class_name_from_id(class_names, class_id)
        objects.extend(
            _component_objects(mask == class_id, class_id, class_name,
                               min_area))
    return objects


def _load_class_id_mask(mask_path, target_shape):
    with PILImage.open(mask_path) as mask_image:
        mask = np.asarray(mask_image)
    if mask.ndim == 3:
        if np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(
                mask[..., 0], mask[..., 2]):
            mask = mask[..., 0]
        else:
            raise ValueError(
                'Ground-truth mask {} is RGB rather than a class-id mask.'.
                format(mask_path))
    if mask.ndim != 2:
        raise ValueError('Unsupported ground-truth mask shape {} for {}.'.format(
            mask.shape, mask_path))
    height, width = target_shape
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def _labelme_objects(json_path, target_shape, class_names, min_area):
    with open(json_path, 'r', encoding='utf-8-sig') as json_file:
        data = json.load(json_file)

    image_height = float(data.get('imageHeight') or target_shape[0])
    image_width = float(data.get('imageWidth') or target_shape[1])
    scale_x = target_shape[1] / image_width if image_width else 1.0
    scale_y = target_shape[0] / image_height if image_height else 1.0
    name_to_id = {
        str(name).strip(): class_id
        for class_id, name in enumerate(class_names or [])
    }
    objects = []
    for shape in data.get('shapes', []):
        class_name = str(shape.get('label', '')).strip()
        points = shape.get('points', [])
        if not class_name or len(points) < 2:
            continue
        contour = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        contour[:, 0] *= scale_x
        contour[:, 1] *= scale_y
        if shape.get('shape_type') == 'rectangle' and len(contour) == 2:
            x1, y1 = contour[0]
            x2, y2 = contour[1]
            contour = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                 dtype=np.float32)
        if len(contour) < 3:
            continue
        area = int(round(abs(cv2.contourArea(contour))))
        if area < min_area:
            continue
        contour = np.rint(contour).astype(np.int32).reshape(-1, 1, 2)
        x, y, width, height = cv2.boundingRect(contour)
        objects.append({
            'class_id': name_to_id.get(class_name, -1),
            'class_name': class_name,
            'area': area,
            'bbox': (x, y, width, height),
            'contour': contour,
        })
    return objects


def _ground_truth_objects(label_path, target_shape, class_names,
                          background_id, ignore_index, min_area):
    if not label_path:
        return None
    if not os.path.isfile(label_path):
        raise FileNotFoundError(
            'Ground-truth label does not exist: {}'.format(label_path))
    if os.path.splitext(label_path)[1].lower() == '.json':
        return _labelme_objects(label_path, target_shape, class_names,
                                min_area)
    mask = _load_class_id_mask(label_path, target_shape)
    return segmentation_objects(
        mask,
        class_names=class_names,
        background_id=background_id,
        ignore_index=ignore_index,
        min_area=min_area)


def _unique_class_names(objects):
    return sorted({item['class_name'] for item in objects})


def _wrap_summary(prefix, names, max_width, font, font_scale, thickness):
    values = names or ['OK']
    lines = []
    current = prefix
    for value in values:
        separator = '' if current == prefix else ', '
        candidate = current + separator + value
        width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if width <= max_width or current == prefix:
            current = candidate
        else:
            lines.append(current)
            current = '    ' + value
    lines.append(current)
    return lines


def _put_text_box(image, text, position, color, font_scale, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    x = max(0, min(x, image.shape[1] - text_size[0] - 6))
    y = max(text_size[1] + 6, min(y, image.shape[0] - baseline - 4))
    cv2.rectangle(image, (x, y - text_size[1] - 5),
                  (x + text_size[0] + 5, y + baseline + 3), (0, 0, 0),
                  cv2.FILLED)
    cv2.putText(image, text, (x + 2, y), font, font_scale, color, thickness,
                cv2.LINE_AA)


def _draw_objects(image, objects, prefix, color, y_offset, font_scale,
                  thickness):
    for item in objects:
        contour = item['contour'].copy()
        contour[:, :, 1] += y_offset
        cv2.polylines(image, [contour], True, color, thickness, cv2.LINE_AA)
        x, y, _, _ = item['bbox']
        _put_text_box(image, '{}:{}'.format(prefix, item['class_name']),
                      (x, y + y_offset - 7), color, font_scale, thickness)


def annotate_segmentation_classes(image,
                                  prediction,
                                  label_path=None,
                                  class_names=None,
                                  background_id=0,
                                  ignore_index=255,
                                  min_area=20,
                                  use_multilabel=False):
    """Add GT and predicted defect types to a blended prediction image.

    A dark header summarizes all present defect types.  Ground-truth objects
    are outlined in green and predicted objects in yellow.  ``label_path`` may
    point to either a class-id mask (including a palette PNG) or a LabelMe JSON
    file.
    """
    if min_area < 1:
        raise ValueError('min_area must be at least 1.')
    annotated_base = np.asarray(image).copy()
    if annotated_base.ndim != 3 or annotated_base.shape[2] != 3:
        raise ValueError('Annotation expects a BGR image with three channels.')
    height, width = annotated_base.shape[:2]

    pred = np.asarray(prediction)
    if not use_multilabel:
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            raise ValueError(
                'Prediction must be a 2-D class-id mask, but got {}.'.format(
                    pred.shape))
        if pred.shape != (height, width):
            pred = cv2.resize(
                pred, (width, height), interpolation=cv2.INTER_NEAREST)
    pred_objects = segmentation_objects(
        pred,
        class_names=class_names,
        background_id=background_id,
        ignore_index=ignore_index,
        min_area=min_area,
        use_multilabel=use_multilabel)
    gt_objects = _ground_truth_objects(label_path, (height, width), class_names,
                                       background_id, ignore_index, min_area)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(1.2, width / 1800.0))
    component_font_scale = max(0.5, font_scale * 0.8)
    thickness = max(1, int(round(font_scale * 2)))
    padding = max(10, int(round(font_scale * 16)))
    line_height = max(24, int(round(font_scale * 34)))
    max_text_width = max(100, width - padding * 2)

    if gt_objects is None:
        gt_lines = ['GT: N/A']
    else:
        gt_lines = _wrap_summary('GT: ', _unique_class_names(gt_objects),
                                 max_text_width, font, font_scale, thickness)
    pred_lines = _wrap_summary('Pred: ', _unique_class_names(pred_objects),
                               max_text_width, font, font_scale, thickness)
    header_height = padding * 2 + line_height * (len(gt_lines) +
                                                 len(pred_lines))
    canvas = np.full(
        (height + header_height, width, 3), (35, 35, 35), dtype=np.uint8)
    canvas[header_height:] = annotated_base

    gt_color = (0, 255, 0)
    pred_color = (0, 255, 255)
    text_y = padding + line_height - 8
    for line in gt_lines:
        cv2.putText(canvas, line, (padding, text_y), font, font_scale, gt_color,
                    thickness, cv2.LINE_AA)
        text_y += line_height
    for line in pred_lines:
        cv2.putText(canvas, line, (padding, text_y), font, font_scale,
                    pred_color, thickness, cv2.LINE_AA)
        text_y += line_height
    cv2.line(canvas, (0, header_height - 1), (width, header_height - 1),
             (180, 180, 180), 1)

    if gt_objects is not None:
        _draw_objects(canvas, gt_objects, 'GT', gt_color, header_height,
                      component_font_scale, thickness)
    _draw_objects(canvas, pred_objects, 'P', pred_color, header_height,
                  component_font_scale, thickness)
    return canvas


def get_pseudo_color_map(pred, color_map=None, use_multilabel=False):
    """
    Get the pseudo color image.

    Args:
        pred (numpy.ndarray): the origin predicted image.
        color_map (list, optional): the palette color map. Default: None,
            use paddleseg's default color map.
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.

    Returns:
        (numpy.ndarray): the pseduo image.
    """
    if use_multilabel:
        bg_pred = (pred.sum(axis=0, keepdims=True) == 0).astype('int32')
        pred = np.concatenate([bg_pred, pred], axis=0)
        gray_idx = np.arange(pred.shape[0]).astype(np.uint8)
        pred = (pred * gray_idx[:, None, None]).sum(axis=0)
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


def paste_images(image_list):
    """
    Paste all image to a image.
    Args:
        image_list (List or Tuple): The images to be pasted and their size are the same.
    Returns:
        result_img (PIL.Image): The pasted image.
    """
    assert isinstance(image_list,
                      (list, tuple)), "image_list should be a list or tuple"
    assert len(
        image_list) > 1, "The length of image_list should be greater than 1"

    pil_img_list = []
    for img in image_list:
        if isinstance(img, str):
            assert os.path.exists(img), "The image is not existed: {}".format(
                img)
            img = PILImage.open(img)
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            img = PILImage.fromarray(img)
        pil_img_list.append(img)

    sample_img = pil_img_list[0]
    size = sample_img.size
    for img in pil_img_list:
        assert size == img.size, "The image size in image_list should be the same"

    width, height = sample_img.size
    result_img = PILImage.new(sample_img.mode,
                              (width * len(pil_img_list), height))
    for i, img in enumerate(pil_img_list):
        result_img.paste(img, box=(width * i, 0))

    return result_img
