#! -*- coding: utf-8 -*-
# 自定义层

import numpy as np

from bert4keras3.backend import keras, ops, is_tf_keras,K,tf
from bert4keras3.backend import align, sequence_masking
from bert4keras3.backend import recompute_grad,int_shape
from bert4keras3.backend import attention_normalize,divide_no_nan
from bert4keras3.backend import sinusoidal_embeddings
from bert4keras3.backend import apply_rotary_position_embeddings
from keras import initializers, activations
from keras.layers import *


def integerize_shape(func):
    """装饰器，保证input_shape一定是int或None
    """
    def convert(item):
        if hasattr(item, '__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item, 'value'):
            return item.value
        else:
            return item

    def new_func(self, input_shape):
        input_shape = convert(input_shape)
        return func(self, input_shape)

    return new_func


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
            mask = ops.cast(mask, K.floatx())
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


class Embedding(keras.layers.Embedding):
    """拓展Embedding层
    """
    def compute_mask(self, inputs, mask=None):
        """为了适配T5，保证第一个token不被mask
        """
        if ops.ndim(inputs) == 2:
            mask = super(Embedding, self).compute_mask(inputs, mask)
            if mask is not None:
                mask1 = ops.ones_like(mask[:, :1], dtype='bool')
                mask2 = mask[:, 1:]
                return ops.concatenate([mask1, mask2], 1)
        else:
            return mask

    def call(self, inputs, mode='embedding'):
        """新增mode参数，可以为embedding或dense。如果为embedding，
        则等价于普通Embedding层；如果为dense，则等价于无bias的Dense层。
        """
        if mode == 'embedding':
            return super(Embedding, self).call(inputs)
        else:
            kernel = ops.transpose(self.embeddings)
            return ops.dot(inputs, kernel)

    def compute_output_shape(self, input_shape):
        """关于判据，本来是通过缓存call时的mode参数来判断的，但是后来发现
        Keras在使用compute_output_shape的时候不一定配套调用了call函数，
        所以缓存的mode可能是不准的，因此只能出此下策。
        """
        if len(input_shape) == 2:
            return super(Embedding, self).compute_output_shape(input_shape)
        else:
            return input_shape[:2] + (int_shape(self.embeddings)[0],)


class ScaleOffset(Layer):
    """简单的仿射变换层（最后一维乘上gamma向量并加上beta向量）
    说明：1、具体操作为最后一维乘上gamma向量并加上beta向量；
         2、如果直接指定scale和offset，那么直接常数缩放和平移；
         3、hidden_*系列参数仅为有条件输入时(conditional=True)使用，
            用于通过外部条件控制beta和gamma。
    """
    def __init__(
        self,
        scale=True,
        offset=True,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs
    ):
        super(ScaleOffset, self).__init__(**kwargs)
        self.scale = scale
        self.offset = offset
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(ScaleOffset, self).build(input_shape)
        
        if self.conditional:
            input_shape = input_shape[0]

        if self.offset is True:
            self.beta = self.add_weight(
                name='beta', shape=(input_shape[-1],), initializer='zeros'
            )
        if self.scale is True:
            self.gamma = self.add_weight(
                name='gamma', shape=(input_shape[-1],), initializer='ones'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.offset is not False and self.offset is not None:
                self.beta_dense = Dense(
                    units=input_shape[-1],
                    use_bias=False,
                    kernel_initializer='zeros'
                )
            if self.scale is not False and self.scale is not None:
                self.gamma_dense = Dense(
                    units=input_shape[-1],
                    use_bias=False,
                    kernel_initializer='zeros'
                )

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            return mask if mask is None else mask[0]
        else:
            return mask

    @recompute_grad
    def call(self, inputs):
        """如果带有条件，则默认以list为输入，第二个是条件
        """
        if self.conditional:
            inputs, conds = inputs
            if self.hidden_units is not None:
                conds = self.hidden_dense(conds)
            conds = align(conds, [0, -1], ops.ndim(inputs))

        if self.scale is not False and self.scale is not None:
            gamma = self.gamma if self.scale is True else self.scale
            if self.conditional:
                gamma = gamma + self.gamma_dense(conds)
            inputs = inputs * gamma

        if self.offset is not False and self.offset is not None:
            beta = self.beta if self.offset is True else self.offset
            if self.conditional:
                beta = beta + self.beta_dense(conds)
            inputs = inputs + beta

        return inputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'scale': self.scale,
            'offset': self.offset,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer':
                initializers.serialize(self.hidden_initializer),
        }
        base_config = super(ScaleOffset, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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


class MultiHeadAttention(Layer):
    """多头注意力机制
    """
    def __init__(
        self,
        heads,
        head_size,
        out_dim=None,
        key_size=None,
        use_bias=True,
        normalization='softmax',
        attention_scale=True,
        attention_dropout=None,
        return_attention_scores=False,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
            
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        if self.attention_dropout:
            self.dropout=Dropout(self.attention_dropout)
        super(MultiHeadAttention, self).build(input_shape)
    @recompute_grad
    def call(self, inputs, mask=None, **kwargs):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        """
        q, k, v = inputs[:3]
        q_mask, v_mask = None, None
        if mask is not None:
            q_mask, v_mask = mask[0], mask[2]
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = ops.reshape(qw, (self.heads, self.key_size), -1)
        kw = ops.reshape(kw, (self.heads, self.key_size), -1)
        vw = ops.reshape(vw, (self.heads, self.head_size), -1)
        # Attention
        qkv_inputs = [qw, kw, vw] + inputs[3:]
        qv_masks = [q_mask, v_mask]
        o, a = self.pay_attention_to(qkv_inputs, qv_masks, **kwargs)
        # 完成输出
        o = self.o_dense(ops.flatten(o, 2))
        # 返回结果
        if self.return_attention_scores:
            return [o, a]
        else:
            return o

    def pay_attention_to(self, inputs, mask=None, **kwargs):
        """实现标准的乘性多头注意力
        a_bias: 对attention矩阵的bias。
                不同的attention bias对应不同的应用。
        p_bias: 在attention里的位置偏置。
                一般用来指定相对位置编码的种类。
        说明: 这里单独分离出pay_attention_to函数，是为了方便
              继承此类来定义不同形式的attention；此处要求
              返回o.shape=(batch_size, seq_len, heads, head_size)。
        """
        (qw, kw, vw), n = inputs[:3], 3
        q_mask, v_mask = mask
        a_bias, p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        if a_bias:
            
            a_bias = inputs[n]
            
            n += 1
        if p_bias == 'rotary':
            qw, kw = apply_rotary_position_embeddings(inputs[n], qw, kw)
        # Attention
        a = ops.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 处理位置编码
        if p_bias == 'typical_relative':
            position_bias = inputs[n]
            a = a + ops.einsum('bjhd,jkd->bhjk', qw, position_bias)
        elif p_bias == 't5_relative':
            position_bias = ops.transpose(inputs[n], (2, 0, 1))
            a = a + ops.expand_dims(position_bias, 0)
        # Attention（续）
        if self.attention_scale:
            a = a / self.key_size**0.5
        if a_bias is not None and ops.ndim(a_bias) == 3:
            a_bias = align(a_bias, [0, -2, -1], ops.ndim(a))
        A = attention_normalize(a, v_mask, -1, self.normalization, a_bias)
        
        if self.attention_dropout:
            A = self.dropout(A)
        # 完成输出
        o = ops.einsum('bhjk,bkhd->bjhd', A, vw)
        if p_bias == 'typical_relative':
            o = o + ops.einsum('bhjk,jkd->bjhd', A, position_bias)
        return o, a

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        if self.return_attention_scores:
            a_shape = (
                input_shape[0][0], self.heads, input_shape[0][1],
                input_shape[1][1]
            )
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                return [mask[0], None]
            else:
                return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'normalization': self.normalization,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GatedAttentionUnit(Layer):
    """门控注意力单元
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
    说明：没有加入加性相对位置编码，个人认为是不必要的；如果觉得有必要，
         可以自行通过a_bias传入。
    """
    def __init__(
        self,
        units,
        key_size,
        activation='swish',
        use_bias=True,
        normalization='squared_relu',
        self_attention=True,
        attention_scale=True,
        attention_dropout=None,
        kernel_initializer='glorot_uniform',
        factorization=False,#是否把V做分解
        **kwargs
    ):
        super(GatedAttentionUnit, self).__init__(**kwargs)
        self.units = units
        self.key_size = key_size
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.normalization = normalization
        self.self_attention = self_attention
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.factorization = factorization
    @integerize_shape
    def build(self, input_shape):
        super(GatedAttentionUnit, self).build(input_shape)
        hidden_size = input_shape[-1]
        if isinstance(hidden_size, (list, tuple)):
            hidden_size = input_shape[0][-1]
        if self.self_attention:
            self.i_dense = Dense(
                units=2 * self.units + self.key_size,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            self.q_scaleoffset = ScaleOffset(offset=self.use_bias)
            self.k_scaleoffset = ScaleOffset(offset=self.use_bias)
        else:
            
            self.u_dense = Dense(
                units=self.units ,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            self.q_dense = Dense(
                units=self.key_size,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            self.k_dense = Dense(
                units=self.key_size,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            if self.factorization:
                self.v_dense = Dense(
                    units=self.key_size,
                    activation='linear',
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer
                )
                self.vW_dense = Dense(
                    units=self.units,
                    activation='linear',
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer
                )
            else:
                self.v_dense = Dense(
                    units=self.units,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer
                )
        self.o_dense = Dense(
            units=hidden_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
            
        if self.attention_dropout:
            self.dropout=Dropout(self.attention_dropout)
    @recompute_grad
    def call(self, inputs, mask=None, a_bias=None, p_bias=None):
        
        if not isinstance(inputs, list):
            inputs, mask = [inputs], [mask]
        if self.self_attention:
            x, n = inputs[0], 1
        else:
            
            (q, k, v), n = inputs[:3], 3
        mask = None if mask is None else mask[0]
        if a_bias:
            a_bias = inputs[n]
            n += 1
        # 投影变换
        if self.self_attention:
            x = self.i_dense(x)
            u, v, qk = x[:,:,:self.units],x[:,:,self.units:self.units*2],x[:,:,self.units*2:]
            q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)
        else:
            u, q = self.u_dense(q),self.q_dense(q)
            k, v = self.k_dense(k), self.v_dense(v)
        # 加入RoPE
        if p_bias == 'rotary':
            q, k = apply_rotary_position_embeddings(inputs[n], q, k)
        # Attention
        a = ops.einsum('bmd,bnd->bmn', q, k)
        if self.attention_scale:
            a = a / self.key_size**0.5
        A = attention_normalize(a, mask, -1, self.normalization, a_bias)
        if self.attention_dropout:
            A = self.dropout(A)
        # 计算输出
        try:
            z=ops.einsum('bmn,bnd->bmd', A, v)
        except:
            pass
        if self.self_attention==False and self.factorization:
            z = self.vW_dense(z)
        o = self.o_dense(u * z)
        return o

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            return mask[0]
        else:
            return mask

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)):
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'units': self.units,
            'key_size': self.key_size,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'normalization': self.normalization,
            'self_attention': self.self_attention,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(GatedAttentionUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(ScaleOffset):
    """(Conditional) Layer Normalization
    """
    def __init__(
        self, zero_mean=True, unit_variance=True, epsilon=None, **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.zero_mean = zero_mean
        self.unit_variance = unit_variance
        self.epsilon = epsilon or K.epsilon()

    @recompute_grad
    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是条件
        """
        
        if self.conditional:
            inputs, conds = inputs

        if self.zero_mean:
            mean = ops.mean(inputs, axis=-1, keepdims=True)
            inputs = inputs - mean
            
        if self.unit_variance:
            variance = ops.mean(ops.square(inputs), axis=-1, keepdims=True)
            
            inputs = divide_no_nan(
                inputs, ops.sqrt(variance + self.epsilon)
            )
        if self.conditional:
            inputs = [inputs, conds]

        return super(LayerNormalization, self).call(inputs)

    def get_config(self):
        config = {
            'zero_mean': self.zero_mean,
            'unit_variance': self.unit_variance,
            'epsilon': self.epsilon,
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionEmbedding(Layer):
    """定义可训练的位置Embedding
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        hierarchical=None,
        embeddings_initializer='zeros',
        custom_position_ids=False,
        **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if 'int' not in K.dtype(position_ids):
                position_ids = ops.cast(position_ids, 'int32')
        else:
            input_shape = ops.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = ops.arange(0, seq_len, dtype='int32')[None]

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = ops.take(embeddings, position_ids // self.input_dim)
            embeddings_y = ops.take(embeddings, position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = ops.take(self.embeddings, position_ids)
            else:
                embeddings = self.embeddings[None, :seq_len]

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = ops.tile(embeddings, [batch_size, 1, 1])
            return ops.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SinusoidalPositionEmbedding(Layer):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self,
        output_dim,
        merge_mode='add',
        custom_position_ids=False,
        **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if 'float' not in K.dtype(position_ids):
                position_ids = ops.cast(position_ids, K.floatx())
        else:
            input_shape = ops.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = ops.arange(0, seq_len, dtype=K.floatx())[None]

        embeddings = sinusoidal_embeddings(position_ids, self.output_dim)

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = ops.tile(embeddings, [batch_size, 1, 1])
            return ops.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RelativePositionEmbedding(Layer):
    """相对位置编码
    来自论文：https://arxiv.org/abs/1803.02155
    """
    def __init__(
        self, input_dim, output_dim, embeddings_initializer='zeros', **kwargs
    ):
        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)

    def build(self, input_shape):
        super(RelativePositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
        )

    def call(self, inputs):
        pos_ids = self.compute_position_ids(inputs)
        return ops.take(self.embeddings, pos_ids,axis=0)

    def compute_position_ids(self, inputs):
        q, v = inputs
        # 计算位置差
        q_idxs = ops.arange(0, ops.shape(q)[1], dtype='int32')
        q_idxs = ops.expand_dims(q_idxs, 1)
        v_idxs = ops.arange(0, ops.shape(v)[1], dtype='int32')
        v_idxs = ops.expand_dims(v_idxs, 0)
        pos_ids = v_idxs - q_idxs
        # 后处理操作
        max_position = (self.input_dim - 1) // 2
        pos_ids = ops.clip(pos_ids, -max_position, max_position)
        pos_ids = pos_ids + max_position
        return pos_ids

    def compute_output_shape(self, input_shape):
        return (None, None, self.output_dim)

    def compute_mask(self, inputs, mask):
        if mask!=None:
            return mask[0]
        return mask

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RelativePositionEmbeddingT5(RelativePositionEmbedding):
    """Google T5的相对位置编码
    来自论文：https://arxiv.org/abs/1910.10683
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        max_distance=128,
        bidirectional=True,
        embeddings_initializer='zeros',
        **kwargs
    ):
        super(RelativePositionEmbeddingT5,
              self).__init__(input_dim, output_dim, **kwargs)
        self.max_distance = max_distance
        self.bidirectional = bidirectional

    def compute_position_ids(self, inputs):
        """T5的相对位置分桶（直接翻译自官方T5源码）
        """
        q, v = inputs
        # 计算位置差
        q_idxs = ops.arange(0, ops.shape(q)[1], dtype='int32')
        q_idxs = ops.expand_dims(q_idxs, 1)
        v_idxs = ops.arange(0, ops.shape(v)[1], dtype='int32')
        v_idxs = ops.expand_dims(v_idxs, 0)
        pos_ids = v_idxs - q_idxs
        # 后处理操作
        num_buckets, max_distance = self.input_dim, self.max_distance
        ret = 0
        n = -pos_ids
        if self.bidirectional:
            num_buckets //= 2
            ret += ops.cast(ops.less(n, 0), 'int32') * num_buckets
            n = ops.absolute(n)
        else:
            n = ops.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = ops.less(n, max_exact)
        val_if_large = max_exact + ops.cast(
            ops.log(ops.cast(n, K.floatx()) / max_exact) /
            np.log(max_distance / max_exact) * (num_buckets - max_exact),
            'int32',
        )
        val_if_large = ops.minimum(val_if_large, num_buckets - 1)
        ret += ops.where(is_small, n, val_if_large)
        return ret

    def get_config(self):
        config = {
            'max_distance': self.max_distance,
            'bidirectional': self.bidirectional,
        }
        base_config = super(RelativePositionEmbeddingT5, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(Layer):
    """FeedForward层
    如果activation不是一个list，那么它就是两个Dense层的叠加；如果activation是
    一个list，那么第一个Dense层将会被替换成门控线性单元（Gated Linear Unit）。
    参考论文: https://arxiv.org/abs/2002.05202
    """
    def __init__(
        self,
        units,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        if not isinstance(activation, list):
            activation = [activation]
        self.activation = [activations.get(act) for act in activation]
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]

        for i, activation in enumerate(self.activation):
            i_dense = Dense(
                units=self.units,
                activation=activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            setattr(self, 'i%s_dense' % i, i_dense)

        self.o_dense = Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs):
        x = self.i0_dense(inputs)
        for i in range(1, len(self.activation)):
            x = x * getattr(self, 'i%s_dense' % i)(inputs)
        x = self.o_dense(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': [
                activations.serialize(act) for act in self.activation
            ],
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GlobalPointer(Layer):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """
    def __init__(
        self,
        heads,
        head_size,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
        kernel_initializer='lecun_normal',
        **kwargs
    ):
        super(GlobalPointer, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.use_bias = use_bias
        self.tril_mask = tril_mask
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(GlobalPointer, self).build(input_shape)
        self.dense = Dense(
            units=self.head_size * self.heads * 2,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def compute_mask(self, inputs, mask=None):
        return None

    @recompute_grad
    def call(self, inputs, mask=None):
        # 输入变换
        inputs = self.dense(inputs)
        inputs = ops.split(inputs, self.heads, axis=-1)
        inputs = ops.stack(inputs, axis=-2)
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            qw, kw = apply_rotary_position_embeddings(pos, qw, kw)
        # 计算内积
        logits = ops.einsum('bmhd,bnhd->bhmn', qw, kw) / self.head_size**0.5
        # 排除下三角
        if self.tril_mask:
            tril_mask = ops.triu(ops.ones_like(logits[0, 0]))
            tril_mask = ops.cast(tril_mask, 'bool')
        else:
            tril_mask = None
        # 返回最终结果
        return sequence_masking(logits, mask, -np.inf, [2, 3], tril_mask)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.heads, input_shape[1], input_shape[1])

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'RoPE': self.RoPE,
            'use_bias': self.use_bias,
            'tril_mask': self.tril_mask,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(GlobalPointer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EfficientGlobalPointer(GlobalPointer):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    """
    def build(self, input_shape):
        self.p_dense = Dense(
            units=self.head_size * 2,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.q_dense = Dense(
            units=self.heads * 2,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.built = True

    @recompute_grad
    def call(self, inputs, mask=None):
        # 输入变换
        inputs = self.p_dense(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            qw, kw = apply_rotary_position_embeddings(pos, qw, kw)
        # 计算内积
        logits = ops.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5
        bias = ops.einsum('bnh->bhn', self.q_dense(inputs)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        # 排除下三角
        if self.tril_mask:
            tril_mask = ops.triu(ops.ones_like(logits[0, 0]))
            tril_mask = ops.cast(tril_mask, 'bool')
        else:
            tril_mask = None
        # 返回最终结果
        return sequence_masking(logits, mask, -np.inf, [2, 3], tril_mask)


class Loss(Layer):
    """特殊的层，用来定义复杂loss
    """
    def __init__(self, output_axis=None, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.output_axis = output_axis

    def call(self, inputs, mask=None):
        loss = self.compute_loss(inputs, mask)
        self.add_loss(loss, inputs=inputs)
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
