#! -*- coding: utf-8 -*-
# 自定义层

import numpy as np

from bert4keras3.backend import keras, ops, is_tf_keras,K,tf
from bert4keras3.backend import align, sequence_masking,backlib
from bert4keras3.backend import recompute_grad,int_shape,integerize_shape
from keras import initializers, activations
from keras.layers import *
if keras.__version__<'3.0':
    from tensorflow import random
else:
    from keras import random
from bert4keras3.Layers_add.sampler import *
from bert4keras3.Layers_add.Embeddings import *
from bert4keras3.Layers_add.GP import *
from bert4keras3.Layers_add.LayerNorms import *
from bert4keras3.Layers_add.Attentions import *
from bert4keras3.Layers_add.FFN import *


if (not is_tf_keras) and keras.__version__ < '2.3':
    import tensorflow as tf
    class Layer(keras.layers.Layer):
        """重新定义Layer，赋予“层中层”功能
        （仅keras 2.3以下版本需要）
        """
        def __init__(self, **kwargs):
            super(Layer, self).__init__(**kwargs)
            self.supports_masking = True  # 本项目的自定义层均可mask

        def __setattr__(self, name, value):
            if isinstance(value, keras.layers.Layer):
                if not hasattr(self, '_layers'):
                    self._layers = []
                if value not in self._layers:
                    self._layers.append(value)
            super(Layer, self).__setattr__(name, value)

        @property
        def trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            if trainable:
                trainable_weights = super(Layer, self).trainable_weights[:]
                for l in getattr(self, '_layers', []):
                    trainable_weights += l.trainable_weights
                return trainable_weights
            else:
                return []

        @property
        def non_trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            non_trainable_weights = super(Layer, self).non_trainable_weights[:]
            for l in getattr(self, '_layers', []):
                if trainable:
                    non_trainable_weights += l.non_trainable_weights
                else:
                    non_trainable_weights += l.weights
            return non_trainable_weights

    if keras.__version__ < '2.2.5':

        import inspect

        class Model(keras.models.Model):
            """重新定义Model，整合fit和fit_generator
            """
            def fit(self, x=None, *args, **kwargs):
                if inspect.isgenerator(x):
                    return self.fit_generator(x, *args, **kwargs)
                else:
                    return super(Model, self).fit(x, *args, **kwargs)

        keras.models.Model = Model

else:

    class Layer(keras.layers.Layer):
        def __init__(self, **kwargs):
            super(Layer, self).__init__(**kwargs)
            self.supports_masking = True  # 本项目的自定义层均可mask


if ((not is_tf_keras) or tf.__version__ < '1.15') and keras.__version__ < '3':

    if not is_tf_keras:
        NodeBase = keras.engine.base_layer.Node
    else:
        from tensorflow.python.keras.engine import base_layer
        NodeBase = base_layer.Node

    class Node(NodeBase):
        """修改Node来修复keras下孪生网络的bug
        注意：这是keras的bug，并不是bert4keras的bug，但keras已经不更新了，
             所以只好在这里进行修改。tf 1.15+自带的keras已经修改了这个
             bug。
        """
        @property
        def arguments(self):
            return self._arguments.copy()

        @arguments.setter
        def arguments(self, value):
            self._arguments = value or {}

    if not is_tf_keras:
        keras.engine.base_layer.Node = Node
    else:
        base_layer.Node = Node


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """重新定义GlobalAveragePooling1D，支持序列长度为None
    """
    def call(self, inputs, mask=None):
        axis = 1 if self.data_format == 'channels_last' else 2
        if mask is not None:
            mask = ops.cast(mask, keras.mixed_precision.dtype_policy().name)
            mask = mask[..., None] if axis == 1 else mask[:, None]
            return ops.sum(inputs * mask, axis=axis) / ops.sum(mask, axis=axis)
        else:
            return ops.mean(inputs, axis=axis)


class GlobalMaxPooling1D(keras.layers.GlobalMaxPooling1D):
    """重新定义GlobalMaxPooling1D，支持mask
    """
    def __init__(self, data_format='channels_last', **kwargs):
        super(GlobalMaxPooling1D, self).__init__(data_format, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        axis = 1 if self.data_format == 'channels_last' else 2
        inputs = sequence_masking(inputs, mask, -np.inf, axis)
        return ops.max(inputs, axis=axis)

    def compute_mask(self, inputs, mask=None):
        return None


# 直接覆盖原对象
keras.layers.GlobalAveragePooling1D = GlobalAveragePooling1D
keras.layers.GlobalMaxPooling1D = GlobalMaxPooling1D

class Concatenate1D(Layer):
    """1维序列拼接层
    说明：本来该功能可以直接通过Concatenate层来实现，无奈Keras
         自带的Concatenate层的compute_mask写得不合理，导致一个
         带mask的序列与一个不带mask的序列拼接会报错，因此干脆
         自己重写一个好了。
    """
    def call(self, inputs):
        return ops.concatenate(inputs, axis=1)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            masks = []
            for i, m in enumerate(mask):
                if m is None:
                    m = ops.ones_like(inputs[i][..., 0], dtype='bool')
                masks.append(m)
            return ops.concatenate(masks, axis=1)

    def compute_output_shape(self, input_shape):
        if all([shape[1] for shape in input_shape]):
            seq_len = sum([shape[1] for shape in input_shape])
            return (input_shape[0][0], seq_len, input_shape[0][2])
        else:
            return (input_shape[0][0], None, input_shape[0][2])


class BatchSplit(Layer):
    """将第一维进行分割
    主要是用于自行实现多卡数据并行。
    """
    def __init__(self, parts, **kwargs):
        super(BatchSplit, self).__init__(**kwargs)
        self.parts = parts

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            return [o for i in mask for o in self.compute_mask(inputs, i)]

        if mask is not None:
            return self.call(mask)
        elif np.ndim(self.parts) > 0:
            return [None] * len(self.parts)
        else:
            return [None] * self.parts

    def call(self, inputs):
        if isinstance(inputs, list):
            return [o for i in inputs for o in self.call(i)]

        outputs = []

        batch_size = ops.shape(inputs)[0]
        if np.ndim(self.parts) > 0:
            batch_size = ops.cast(batch_size, 'float64')
            slices = [
                ops.cast(p * batch_size / sum(self.parts), 'int32')
                for p in np.cumsum(self.parts).astype('float64')
            ]
        else:
            stride = ops.cast(
                ops.ceil(batch_size / self.parts), K.dtype(batch_size)
            )
            slices = [stride * (i + 1) for i in range(self.parts)]

        for i, _ in enumerate(slices):
            if i == 0:
                outputs.append(inputs[:slices[0]])
            elif i == len(slices) - 1:
                outputs.append(inputs[slices[-2]:])
            else:
                outputs.append(inputs[slices[i - 1]:slices[i]])

        return outputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return [
                o for i in input_shape for o in self.compute_output_shape(i)
            ]

        if np.ndim(self.parts) > 0:
            return [input_shape] * len(self.parts)
        else:
            return [input_shape] * self.parts

    def get_config(self):
        config = {
            'parts': self.parts,
        }
        base_config = super(BatchSplit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BatchConcat(Layer):
    """将第一维进行合并
    主要是用于自行实现多卡数据并行。
    """
    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            if all([m is not None for m in mask]):
                return ops.concatenate(mask, 0)

    def call(self, inputs):
        return ops.concatenate(inputs, 0)

    def compute_output_shape(self, input_shape):
        return input_shape[0]



class Loss(Layer):
    """特殊的层，用来定义复杂loss
    """
    def __init__(self, output_axis=None, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.output_axis = output_axis

    def call(self, inputs, mask=None):
        loss = self.compute_loss(inputs, mask)
        self.add_loss(loss)
        if self.output_axis is None:
            return inputs
        elif isinstance(self.output_axis, list):
            return [inputs[i] for i in self.output_axis]
        else:
            return inputs[self.output_axis]

    def compute_loss(self, inputs, mask=None):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        if self.output_axis is None:
            return input_shape
        elif isinstance(self.output_axis, list):
            return [input_shape[i] for i in self.output_axis]
        else:
            return input_shape[self.output_axis]

    def compute_mask(self, inputs, mask):
        if mask is not None:
            if self.output_axis is None:
                return mask
            elif isinstance(self.output_axis, list):
                return [mask[i] for i in self.output_axis]
            else:
                return mask[self.output_axis]

    def get_config(self):
        config = {
            'output_axis': self.output_axis,
        }
        base_config = super(Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def Input(
    shape=None,
    batch_size=None,
    dtype=None,
    sparse=None,
    batch_shape=None,
    name=None,
    tensor=None,
    ):
    if dtype==None:
        dtype='float32'
    return keras.Input(
            shape=shape,
            batch_size=batch_size,
            dtype=dtype,
            sparse=sparse,
            batch_shape=batch_shape,
            name=name,
            tensor=tensor,
            )
keras.layers.Input = Input

custom_objects = {
    'Embedding': Embedding,
    'ScaleOffset': ScaleOffset,
    'Concatenate1D': Concatenate1D,
    'BatchSplit': BatchSplit,
    'BatchConcat': BatchConcat,
    'MultiHeadAttention': MultiHeadAttention,
    'GatedAttentionUnit': GatedAttentionUnit,
    'LayerNormalization': LayerNormalization,
    'PositionEmbedding': PositionEmbedding,
    'SinusoidalPositionEmbedding': SinusoidalPositionEmbedding,
    'RelativePositionEmbedding': RelativePositionEmbedding,
    'RelativePositionEmbeddingT5': RelativePositionEmbeddingT5,
    'FeedForward': FeedForward,
    'GlobalPointer': GlobalPointer,
    'EfficientGlobalPointer': EfficientGlobalPointer,
    'Loss': Loss,
}

keras.utils.get_custom_objects().update(custom_objects)
