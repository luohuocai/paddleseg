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

import os
import contextlib
import tempfile
import random
from urllib.parse import urlparse, unquote

import yaml
import numpy as np
import paddle
import cv2

from paddleseg.utils import logger, seg_env, get_sys_env
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.models.layers.layer_libs import NaiveSyncBatchNorm


def set_seed(seed=None):
    if seed is not None:
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def show_env_info():
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)


def show_cfg_info(config):
    msg = '\n---------------Config Information---------------\n'
    ordered_module = ('batch_size', 'iters', 'train_dataset', 'val_dataset',
                      'optimizer', 'lr_scheduler', 'loss', 'model')
    all_module = set(config.dic.keys())
    for module in ordered_module:
        if module in config.dic:
            module_dic = {module: config.dic[module]}
            msg += str(yaml.dump(module_dic, Dumper=NoAliasDumper))
            all_module.remove(module)
    for module in all_module:
        module_dic = {module: config.dic[module]}
        msg += str(yaml.dump(module_dic, Dumper=NoAliasDumper))
    msg += '------------------------------------------------\n'
    logger.info(msg)


def set_device(device):
    env_info = get_sys_env()
    if 'gpu' in device and env_info['Paddle compiled with cuda'] \
        and env_info['GPUs used']:
        place = device
    elif 'xpu' in device and paddle.is_compiled_with_xpu():
        place = device
    elif 'npu' in device and paddle.is_compiled_with_custom_device('npu'):
        place = device
    elif device in paddle.device.get_all_custom_device_type():
        place = device
    else:
        place = 'cpu'
    paddle.set_device(place)
    logger.info("Set device: {}".format(place))


def convert_sync_batchnorm(model, device):
    # Convert bn to sync_bn when use multi GPUs
    env_info = get_sys_env()
    if device == 'gpu' and env_info['Paddle compiled with cuda'] \
        and env_info['GPUs used'] and paddle.distributed.ParallelEnv().nranks > 1:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Convert bn to sync_bn")
    elif device == "npu" and paddle.distributed.ParallelEnv().nranks > 1:
        model = NaiveSyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Convert bn to sync_bn in NPU Device")
    return model


def set_cv2_num_threads(num_workers):
    # Limit cv2 threads if too many subprocesses are spawned.
    # This should reduce resource allocation and thus boost performance.
    nranks = paddle.distributed.ParallelEnv().nranks
    if nranks >= 8 and num_workers >= 8:
        logger.warning("The number of threads used by OpenCV is " \
            "set to 1 to improve performance.")
        cv2.setNumThreads(1)


@contextlib.contextmanager
def generate_tempdir(directory: str = None, **kwargs):
    '''Generate a temporary directory'''
    directory = seg_env.TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir


def load_entire_model(model, pretrained):
    if pretrained is not None:
        load_pretrained_model(model, pretrained)
    else:
        logger.warning('Weights are not loaded for {} model since the '
                       'path of weights is None'.format(
                           model.__class__.__name__))


def download_pretrained_model(pretrained_model):
    """
    Download pretrained model from url.
    Args:
        pretrained_model (str): the url of pretrained weight
    Returns:
        str: the path of pretrained weight
    """
    assert urlparse(pretrained_model).netloc, "The url is not valid."

    pretrained_model = unquote(pretrained_model)
    savename = pretrained_model.split('/')[-1]
    if not savename.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        savename = pretrained_model.split('/')[-2]
        filename = pretrained_model.split('/')[-1]
    else:
        savename = savename.split('.')[0]
        filename = 'model.pdparams'

    with generate_tempdir() as _dir:
        pretrained_model = download_file_and_uncompress(
            pretrained_model,
            savepath=_dir,
            cover=False,
            extrapath=seg_env.PRETRAINED_MODEL_HOME,
            extraname=savename,
            filename=filename)
        pretrained_model = os.path.join(pretrained_model, filename)
    return pretrained_model


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        logger.info('Loading pretrained model from {}'.format(pretrained_model))

        if urlparse(pretrained_model).netloc:
            pretrained_model = download_pretrained_model(pretrained_model)

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(
                        model_state_dict[k].shape):
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape,
                                model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict),
                model.__class__.__name__))

        else:
            raise ValueError(
                'The pretrained model directory is not Found: {}'.format(
                    pretrained_model))
    else:
        logger.info(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))


def resume(model, optimizer, resume_model):
    if resume_model is not None:
        logger.info('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            ckpt_path = os.path.join(resume_model, 'model.pdopt')
            opti_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
            optimizer.set_state_dict(opti_state_dict)

            iter = resume_model.split('_')[-1]
            iter = int(iter)
            return iter
        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.
                format(resume_model))
    else:
        logger.info('No model needed to resume.')


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 100000))


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
        else:
            image_dir = os.path.dirname(image_path)
            with open(image_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if len(line.split()) > 1:
                        line = line.split()[0]
                    image_list.append(os.path.join(image_dir, line))
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if f.startswith('.'):
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be a path of image, or a file list containing image paths, or a directory including images.'
        )

    if len(image_list) == 0:
        raise RuntimeError(
            'There are not image file in `--image_path`={}'.format(image_path))

    return image_list, image_dir


def get_image_list_with_labels(image_path, label_dir=None):
    """Get images and optional labels used by prediction visualizations.

    PaddleSeg file lists use ``image_path label_path`` on each line.  The
    existing :func:`get_image_list` intentionally discards the second column;
    this helper preserves it.  For a single image or directory input, labels
    are also discovered using the conventions used by defect-evaluation data:
    a same-stem LabelMe JSON or a same-name mask below ``labels``.

    Args:
        image_path (str): An image, an image directory, or a PaddleSeg file
            list.
        label_dir (str, optional): Explicit root containing class-id masks or
            LabelMe JSON files. Labels are matched by relative path first and
            filename/stem second.

    Returns:
        tuple: ``(image_list, image_dir, label_map)``.  ``label_map`` is keyed
        by normalized absolute image path.
    """
    image_list, image_dir = get_image_list(image_path)
    label_map = {}
    valid_suffix = {
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    }

    is_image_file = (os.path.isfile(image_path) and
                     os.path.splitext(image_path)[-1] in valid_suffix)
    is_file_list = os.path.isfile(image_path) and not is_image_file

    if is_file_list:
        list_dir = os.path.dirname(os.path.abspath(image_path))
        with open(image_path, 'r', encoding='utf-8-sig') as file_list:
            for line in file_list:
                columns = line.strip().split()
                if len(columns) < 2:
                    continue
                image_file, label_file = columns[:2]
                if not os.path.isabs(image_file):
                    image_file = os.path.join(list_dir, image_file)
                if not os.path.isabs(label_file):
                    label_file = os.path.join(list_dir, label_file)
                image_key = os.path.normcase(
                    os.path.abspath(os.path.normpath(image_file)))
                label_map[image_key] = os.path.normpath(label_file)

    if os.path.isdir(image_path):
        dataset_root = os.path.abspath(image_path)
        filtered_images = []
        for image_file in image_list:
            relative_parts = os.path.relpath(image_file,
                                             dataset_root).split(os.sep)
            if any(part.lower() == 'labels'
                   for part in relative_parts[:-1]):
                continue
            filtered_images.append(image_file)
        image_list = filtered_images
    elif is_image_file:
        dataset_root = os.path.dirname(os.path.abspath(image_path))
    else:
        dataset_root = os.path.dirname(os.path.abspath(image_path))

    if label_dir is not None:
        label_dir = os.path.abspath(label_dir)
    for image_file in image_list:
        image_key = os.path.normcase(
            os.path.abspath(os.path.normpath(image_file)))
        if image_key in label_map:
            continue
        label_path = None
        if label_dir is not None:
            label_path = _discover_label_in_directory(
                image_file, dataset_root, label_dir)
        if label_path is None:
            label_path = _discover_prediction_label(image_file, dataset_root)
        if label_path is not None:
            label_map[image_key] = label_path

    return image_list, image_dir, label_map


def _discover_prediction_label(image_file, dataset_root):
    """Find a LabelMe JSON or class-id mask associated with an image."""
    image_file = os.path.abspath(image_file)
    dataset_root = os.path.abspath(dataset_root)
    image_dir = os.path.dirname(image_file)
    image_name = os.path.basename(image_file)
    image_stem = os.path.splitext(image_name)[0]
    try:
        relative_image = os.path.relpath(image_file, dataset_root)
    except ValueError:
        relative_image = image_name

    json_candidates = [
        os.path.splitext(image_file)[0] + '.json',
        os.path.join(dataset_root, image_stem + '.json'),
    ]
    mask_candidates = [
        os.path.join(dataset_root, 'labels', relative_image),
        os.path.join(dataset_root, 'labels', image_name),
        os.path.join(image_dir, 'labels', image_name),
    ]
    seen = set()
    for candidate in json_candidates + mask_candidates:
        candidate = os.path.normpath(candidate)
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isfile(candidate):
            return candidate
    return None


def _discover_label_in_directory(image_file, image_root, label_dir):
    """Match an image to a mask in an explicitly supplied label directory."""
    image_file = os.path.abspath(image_file)
    image_root = os.path.abspath(image_root)
    label_dir = os.path.abspath(label_dir)
    image_name = os.path.basename(image_file)
    image_stem = os.path.splitext(image_name)[0]
    try:
        relative_image = os.path.relpath(image_file, image_root)
    except ValueError:
        relative_image = image_name

    relative_stem = os.path.splitext(relative_image)[0]
    candidates = [
        os.path.join(label_dir, relative_stem + '.json'),
        os.path.join(label_dir, image_stem + '.json'),
        os.path.join(label_dir, relative_image),
        os.path.join(label_dir, image_name),
    ]
    for extension in ('.png', '.bmp', '.tif', '.tiff'):
        candidates.extend([
            os.path.join(label_dir, relative_stem + extension),
            os.path.join(label_dir, image_stem + extension),
        ])
    seen = set()
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isfile(candidate):
            return candidate
    return None


class NoAliasDumper(yaml.SafeDumper):

    def ignore_aliases(self, data):
        return True


class CachedProperty(object):
    """
    A property that is only computed once per instance and then replaces itself with an ordinary attribute.

    The implementation refers to https://github.com/pydanny/cached-property/blob/master/cached_property.py .
        Note that this implementation does NOT work in multi-thread or coroutine senarios.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func
        self.__doc__ = getattr(func, '__doc__', '')

    def __get__(self, obj, cls):
        if obj is None:
            return self
        val = self.func(obj)
        # Hack __dict__ of obj to inject the value
        # Note that this is only executed once
        obj.__dict__[self.func.__name__] = val
        return val


def get_in_channels(model_cfg):
    if 'backbone' in model_cfg:
        return model_cfg['backbone'].get('in_channels', None)
    else:
        return model_cfg.get('in_channels', None)


def set_in_channels(model_cfg, in_channels):
    model_cfg = model_cfg.copy()
    if 'backbone' in model_cfg:
        model_cfg['backbone']['in_channels'] = in_channels
    else:
        model_cfg['in_channels'] = in_channels
    return model_cfg
