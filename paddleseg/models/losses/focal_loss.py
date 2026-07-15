# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class FocalLoss(nn.Layer):
    """
    The implement of focal loss.

    The focal loss requires the label is 0 or 1 for now.

    Args:
        alpha (float, list, optional): The alpha of focal loss. alpha is the weight
            of class 1, 1-alpha is the weight of class 0. Default: 0.25
        gamma (float, optional): The gamma of Focal Loss. Default: 2.0
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.EPS = 1e-10

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C, H, W), where C is number of classes.
            label (Tensor): Label tensor, the data type is int64. Shape is (N, H, W),
                where each value is 0 <= label[i] <= C-1.
        Returns:
            (Tensor): The average loss.
        """
        assert logit.ndim == 4, "The ndim of logit should be 4."
        assert logit.shape[1] == 2, "The channel of logit should be 2."
        assert label.ndim == 3, "The ndim of label should be 3."

        class_num = logit.shape[1]  # class num is 2
        logit = paddle.transpose(logit, [0, 2, 3, 1])  # N,C,H,W => N,H,W,C

        mask = label != self.ignore_index  # N,H,W
        label = paddle.where(mask, label, paddle.zeros_like(label))
        mask = paddle.unsqueeze(mask, 3)
        mask = paddle.cast(mask, 'float32')
        mask.stop_gradient = True

        label = F.one_hot(label, class_num)  # N,H,W,C
        label = paddle.cast(label, logit.dtype)
        label.stop_gradient = True

        loss = F.sigmoid_focal_loss(
            logit=logit,
            label=label,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='none')
        loss = loss * mask
        avg_loss = paddle.sum(loss) / (
            paddle.sum(paddle.cast(mask != 0., 'int32')) * class_num + self.EPS)
        return avg_loss


@manager.LOSSES.add_component
class MultiClassFocalLoss(nn.Layer):
    """
    The implement of focal loss for multi class.

    Args:
        alpha (float, list, optional): The alpha of focal loss. alpha is the weight
            of class 1, 1-alpha is the weight of class 0. Default: 0.25
        gamma (float, optional): The gamma of Focal Loss. Default: 2.0
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.EPS = 1e-10

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C, H, W), where C is number of classes.
            label (Tensor): Label tensor, the data type is int64. Shape is (N, H, W),
                where each value is 0 <= label[i] <= C-1.
        Returns:
            (Tensor): The average loss.
        """
        assert logit.ndim == 4, "The ndim of logit should be 4."
        assert label.ndim == 3, "The ndim of label should be 3."

        logit = paddle.transpose(logit, [0, 2, 3, 1])
        label = label.astype('int64')
        ce_loss = F.cross_entropy(
            logit, label, ignore_index=self.ignore_index, reduction='none')

        pt = paddle.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt)**self.gamma) * ce_loss

        mask = paddle.cast(label != self.ignore_index, 'float32')
        focal_loss *= mask
        avg_loss = paddle.mean(focal_loss) / (paddle.mean(mask) + self.EPS)
        return avg_loss


@manager.LOSSES.add_component
class FocalTverskyLoss(nn.Layer):
    """
    Multi-class focal Tversky loss for imbalanced segmentation.

    Args:
        alpha (float, optional): False positive weight. Default: 0.3.
        beta (float, optional): False negative weight. Default: 0.7.
        gamma (float, optional): Focal exponent. Default: 1.33.
        include_background (bool, optional): Whether class 0 is included.
            Default: True.
        present_classes_only (bool, optional): Average only classes present in
            labels. Default: False.
        ignore_index (int64, optional): Label value to ignore. Default: 255.
        smooth (float, optional): Smoothing value. Default: 1.0.
    """

    def __init__(self,
                 alpha=0.3,
                 beta=0.7,
                 gamma=1.33,
                 include_background=True,
                 present_classes_only=False,
                 ignore_index=255,
                 smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.include_background = include_background
        self.present_classes_only = present_classes_only
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = 1e-8

    def forward(self, logit, label):
        assert logit.ndim == 4, "The ndim of logit should be 4."
        assert label.ndim in [3, 4], "The ndim of label should be 3 or 4."

        if label.ndim == 4:
            label = paddle.squeeze(label, axis=1)

        num_classes = logit.shape[1]
        valid_mask = label != self.ignore_index
        safe_label = paddle.where(valid_mask, label, paddle.zeros_like(label))
        safe_label = safe_label.astype('int64')
        label_one_hot = F.one_hot(safe_label, num_classes)
        label_one_hot = paddle.transpose(label_one_hot, [0, 3, 1, 2])
        label_one_hot = paddle.cast(label_one_hot, logit.dtype)

        valid_mask = paddle.unsqueeze(valid_mask, axis=1)
        valid_mask = paddle.cast(valid_mask, logit.dtype)
        valid_mask.stop_gradient = True
        label_one_hot = label_one_hot * valid_mask
        label_one_hot.stop_gradient = True

        prob = F.softmax(logit, axis=1) * valid_mask
        reduce_axes = [0, 2, 3]
        true_pos = paddle.sum(prob * label_one_hot, axis=reduce_axes)
        false_pos = paddle.sum(prob * (1 - label_one_hot), axis=reduce_axes)
        false_neg = paddle.sum((1 - prob) * label_one_hot, axis=reduce_axes)

        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg +
            self.smooth + self.eps)
        loss = paddle.pow(1 - tversky, self.gamma)

        class_mask = paddle.ones([num_classes], dtype=logit.dtype)
        if not self.include_background and num_classes > 1:
            class_mask = paddle.concat(
                [paddle.zeros([1], dtype=logit.dtype), class_mask[1:]])

        if self.present_classes_only:
            present_mask = paddle.sum(label_one_hot, axis=reduce_axes) > 0
            present_mask = paddle.cast(present_mask, logit.dtype)
            class_mask = class_mask * present_mask

        class_mask.stop_gradient = True
        return paddle.sum(loss * class_mask) / (paddle.sum(class_mask) + self.eps)
