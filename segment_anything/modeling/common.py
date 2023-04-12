# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import paddle
import paddle.nn as nn

from typing import Type


class MLPBlock(nn.Layer):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Layer] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Layer):
    def __init__(self, num_channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()

        self.weight = paddle.create_parameter(shape=[num_channels],
                                              dtype='float32',
                                              default_initializer=paddle.nn.initializer.Assign(
                                                  paddle.ones([num_channels])))
        self.bias = paddle.create_parameter(shape=[num_channels],
                                            dtype='float32',
                                            default_initializer=paddle.nn.initializer.Assign(
                                                paddle.zeros([num_channels])))
        self.epsilon = epsilon

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.epsilon)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
