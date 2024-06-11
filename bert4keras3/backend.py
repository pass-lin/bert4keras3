# -*- coding: utf-8 -*-
# 分离后端函数，主要是为了同时兼容原生keras和tf.keras
# 通过设置环境变量TF_KERAS=1来切换tf.keras

import os, sys
from distutils.util import strtobool
import numpy as np
from functools import wraps
is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))
lora_model = strtobool(os.environ.get('ENABLE_LORA', '0'))
load_tf = strtobool(os.environ.get('LOAD_TF', '1'))
if load_tf:
    import tensorflow as tf
#jax使用flash参考https://github.com/nshepperd/flash_attn_jax/releases这里安装flash
enable_flashatt = strtobool(os.environ.get('FLASH_ATTN', '0'))
os.environ["KERAS_BACKEND"]=os.environ.get("KERAS_BACKEND", 'tensorflow')
backlib=os.environ["KERAS_BACKEND"]
if backlib=='tfkeras':
    is_tf_keras = True
    if enable_flashatt:
        raise('tensorflow not support flash-attention')
elif backlib=='torch':
    import torch
    if enable_flashatt:
        from flash_attn import flash_attn_func
        def flash_mha(q,k,v,softmax_scale=None, is_causal=False, window_size=(-1,-1)):
            return flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=is_causal,window_size=window_size)
elif backlib=='jax':
    if enable_flashatt:
        from flash_attn_jax import flash_mha
    import jax  
if is_tf_keras:
    sys.modules['keras'] = tf.keras

import keras
import keras.backend as K
do_recompute = strtobool(os.environ.get('RECOMPUTE', '0'))
use_keras_2 = is_tf_keras or keras.__version__<'3.0'

if use_keras_2:
    
    from tensorflow.python.client import device_lib
    from tensorflow.python.util import nest, tf_inspect
    from tensorflow.python.eager import tape
    from tensorflow.python.ops.custom_gradient import _graph_mode_decorator
    import bert4keras3.ops as ops
    load_variable=tf.train.load_variable
    norm=tf.norm
else:
    from keras import ops
    def recompute_grad(call):
        if not do_recompute:
            return call
        return call
    if backlib=='torch':
        def norm(tensor, ord='euclidean', axis=None, keepdims=None):
            if ord=='euclidean':
                ord=None
            return torch.linalg.norm(tensor, ord, axis, keepdims)  
    elif backlib=='jax':
        def recompute_grad(call):
            if not do_recompute:
                return call
            return jax.checkpoint(call)
        def norm(tensor, ord='euclidean', axis=None, keepdims=None):
            if ord=='euclidean':
                ord=None
            return jax.numpy.linalg.norm(tensor, ord, axis, keepdims)
        
    else:
        def recompute_grad(call):
           if not do_recompute:
               return call
           return tf.recompute_grad(call)
        norm=tf.norm
ops.norm=norm
# 判断是否启用重计算（通过时间换空间）
do_recompute = strtobool(os.environ.get('RECOMPUTE', '0'))

def divide_no_nan(x1,x2):
    y=ops.divide(x1,x2)
    return ops.where(ops.isnan(y),ops.zeros_like(y),y)
def int_shape(x):
    return ops.convert_to_numpy(x.shape)
def gelu_erf(x):
    """基于Erf直接计算的gelu函数
    """
    return 0.5 * x * (1.0 + ops.erf(x / np.sqrt(2.0)))


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
keras.Layer = Layer
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
def set_gelu(version):
    """设置gelu版本
    """
    version = version.lower()
    assert version in ['erf', 'tanh'], 'gelu version must be erf or tanh'
    if version == 'erf':
        keras.utils.get_custom_objects()['gelu'] = gelu_erf
    else:
        keras.utils.get_custom_objects()['gelu'] = ops.gelu
def piecewise_linear(t, schedule, from_zero=True):
    """分段线性函数
    其中schedule是形如{1000: 1, 2000: 0.1}的字典，
    表示 t ∈ [0, 1000]时，输出从0均匀增加至1，而
    t ∈ [1000, 2000]时，输出从1均匀降低到0.1，最后
    t > 2000时，保持0.1不变。
    """
    schedule = sorted(schedule.items())
    if from_zero and schedule[0][0] != 0:
        schedule = [(0, 0.0)] + schedule

    t = ops.cast(t, keras.mixed_precision.dtype_policy().name)
    x = (t * 0 + 1) * schedule[0][1]
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1.0 * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = (t * 0 + 1) * schedule[i][1]
        x = ops.where(t >= t_begin, x, x_begin)

    return x


def search_layer(inputs, name, exclude_from=None):
    """根据inputs和name来搜索层
    说明：inputs为某个层或某个层的输出；name为目标层的名字。
    实现：根据inputs一直往上递归搜索，直到发现名字为name的层为止；
         如果找不到，那就返回None。
    """
    if exclude_from is None:
        exclude_from = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude_from:
        return None
    else:
        exclude_from.add(layer)
        if isinstance(layer, keras.models.Model):
            model = layer
            for layer in model.layers:
                if layer.name == name:
                    return layer
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude_from)
                if layer is not None:
                    return layer



old_reshape=ops.reshape
def reshape(tensor, *args):
    """实现更灵活的reshape
    其中 *args 为 (shape1, axis1, shape2, axis2, ...) 格式，表示将
    维度axis1转换为shape1、维度axis2转换为shape2、...
    """
    if len(args) == 1:
        return old_reshape(tensor, args[0])
    assert len(args) % 2 == 0
    shape = ops.shape(tensor)
    shape = [[s or shape[i]] for i, s in enumerate(int_shape(tensor))]
    for s, i in zip(args[::2], args[1::2]):
        s = list(s)
        assert s.count(-1) <= 1
        if s.count(-1) == 1:
            j = s.index(-1)
            s[j] = -shape[i][0] // ops.prod(s)
        shape[i] = s
    return old_reshape(tensor, [i for s in shape for i in s])


def flatten(tensor, start=None, end=None):
    """将tensor从start到end的维度展平
    """
    start, end = start or 0, end or ops.ndim(tensor)
    shape = ops.shape(tensor)
    shape = [s or shape[i] for i, s in enumerate(int_shape(tensor))]
    if keras.__version__>'3.0' and os.environ["KERAS_BACKEND"] != "jax":
        shape = shape[:start] + [ops.prod(shape[start:end])] + shape[end:]
        return ops.reshape(tensor, shape)
    shape = shape[:start] + [np.prod(shape[start:end])] + shape[end:]
    return ops.reshape(tensor, shape)


def dtype(x):
    try:
        return x.dtype
    except:
        pass
K.dtype=dtype

if use_keras_2:
    def where(cond, x, y):
        """给tf.where加上自动广播
        """
        shape = tf.broadcast_dynamic_shape(K.shape(x), K.shape(y))
        shape = tf.broadcast_dynamic_shape(K.shape(cond), shape)
    
        if dtype(x) is None and dtype(y) is None:
            x = tf.broadcast_to(K.constant(x, dtype=keras.mixed_precision.dtype_policy().name), shape)
            y = tf.broadcast_to(K.constant(y, dtype=keras.mixed_precision.dtype_policy().name), shape)
        elif dtype(x) is None:
            x = tf.broadcast_to(K.constant(x, dtype=dtype(y)), shape)
        elif dtype(y) is None:
            y = tf.broadcast_to(K.constant(y, dtype=dtype(x)), shape)
        else:
            x = tf.broadcast_to(x, shape)
            y = tf.broadcast_to(y, shape)
    
        if dtype(cond) != 'bool':
            cond = K.cast(cond, 'bool')
    
        cond = tf.broadcast_to(cond, shape)
        return tf.where(cond, x, y)
    ops.where = where

def get_sequence_masking(
    x, mask=None, value=0, axis=None, bias=None
):
    if not (mask is None and bias is None):
        if mask is None:
            if K.dtype(bias) == 'bool' or (backlib=='torch' and K.dtype(bias) == torch.bool):
                mask = bias
        else:
            if axis is None:
                axes = [1]
            elif isinstance(axis, list):
                axes = axis
            else:
                axes = [axis]

            axes = [axis if axis >= 0 else ops.ndim(x) + axis for axis in axes]

            if K.dtype(mask) != 'bool':
                mask = ops.cast(mask, 'bool')
            elif backlib=='torch' and K.dtype(bias) == torch.bool:
                mask = ops.cast(mask, torch.bool)

            full_mask = align(mask, [0, axes[0]], ops.ndim(x))
            for axis in axes[1:]:
                full_mask = full_mask & align(mask, [0, axis], ops.ndim(x))

            mask = full_mask
            if K.dtype(bias) == 'bool' or (backlib=='torch' and K.dtype(bias) == torch.bool):
                mask = mask & bias
    if mask is not None:
        mask = ops.cast(mask,dtype='int32')
    return mask
def sequence_masking(
    x, mask=None, value=0, axis=None, bias=None, return_mask=False
):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的bool矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    bias: 额外的偏置项，或者附加的mask；
    return_mask: 是否同时返回对齐后的mask。
    """
    
    if not (mask is None and bias is None):
        if mask is None:
            if K.dtype(bias) == 'bool' or (backlib=='torch' and K.dtype(bias) == torch.bool):
                mask = bias
                x = ops.where(mask, x, value)
            else:
                x = x + bias
        else:
            if axis is None:
                axes = [1]
            elif isinstance(axis, list):
                axes = axis
            else:
                axes = [axis]

            axes = [axis if axis >= 0 else ops.ndim(x) + axis for axis in axes]

            if K.dtype(mask) != 'bool':
                mask = ops.cast(mask, 'bool')
            elif backlib=='torch' and K.dtype(bias) == torch.bool:
                mask = ops.cast(mask, torch.bool)

            full_mask = align(mask, [0, axes[0]], ops.ndim(x))
            for axis in axes[1:]:
                full_mask = full_mask & align(mask, [0, axis], ops.ndim(x))

            mask = full_mask
            if bias is None:
                x = ops.where(mask, x, value)
            elif K.dtype(bias) == 'bool' or (backlib=='torch' and K.dtype(bias) == torch.bool):
                mask = mask & bias
                x = ops.where(mask, x, value)
            else:
                x = ops.where(mask, x + bias, value)

    if return_mask:
        return x, mask
    else:
        return x



def divisible_temporal_padding(x, n):
    """将一维向量序列右padding到长度能被n整除
    """
    r_len = ops.shape(x)[1] % n
    p_len = ops.where(r_len > 0, n - r_len, 0)
    return ops.pad(x, [[0,0],(0, p_len),[0,0]])


def root_mean_square(x, axis=None, keepdims=False):
    """均方根，相当于模长的变体
    """
    return ops.sqrt(ops.mean(ops.square(x), axis=axis, keepdims=keepdims))


def swish(x):
    """swish函数（这样封装过后才有 __name__ 属性）
    """
    return ops.silu(x)


def leaky_relu(x, alpha=0.2):
    """leaky relu函数（这样封装过后才有 __name__ 属性）
    """
    return ops.leaky_relu(x, alpha=alpha)


def attention_normalize(a, mask=None, axis=-1, method='softmax', bias=None):
    """不同的注意力归一化方案
    softmax：常规/标准的指数归一化；
    squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
    softmax_plus：来自 https://kexue.fm/archives/8823 。
    """
    if method == 'softmax-fp32' :
        mask = get_sequence_masking(a, mask, -np.inf, axis,bias)
        ori_dtype = a.dtype
        att_mask = mask
        for i in range(ops.ndim(a)-ops.ndim(mask)):
            att_mask = ops.expand_dims(att_mask,0)
        return ops.cast(keras.layers.Softmax(dtype="float32",axis=axis)(a,mask=att_mask),ori_dtype)
    a, mask = sequence_masking(a, mask, -np.inf, axis, bias, True)
    
    if method == 'softmax' :
        return ops.softmax(a,axis=axis)
    else:
        if mask is None:
            l = ops.cast(ops.shape(a)[-1], keras.mixed_precision.dtype_policy().name)
        else:
            mask = ops.cast(mask, keras.mixed_precision.dtype_policy().name)
            l = ops.sum(mask, axis=axis, keepdims=True)
        if method == 'squared_relu':
            return ops.relu(a)**2 / l
        elif method == 'softmax_plus':
            l = ops.maximum(l, 16)  # 极短序列scale反而不好
            return ops.softmax(a * ops.log(l) / np.log(512), axis=axis)
    return a


def sinusoidal_embeddings(pos, dim, base=10000):
    """计算pos位置的dim维sinusoidal编码
    """
    assert dim % 2 == 0
    indices = ops.arange(0, dim // 2, dtype='float32')
    indices = ops.power(ops.cast(base, dtype='float32'), -2 * indices / dim)
    embeddings = ops.einsum('...,d->...d', pos, indices)
    embeddings = ops.stack([ops.sin(embeddings), ops.cos(embeddings)], axis=-1)
    embeddings = ops.flatten(embeddings, -2)
    return embeddings


class Sinusoidal(keras.initializers.Initializer):
    """Sin-Cos位置向量初始化器
    来自：https://arxiv.org/abs/1706.03762
    """
    def __call__(self, shape, dtype=None):
        """Sin-Cos形式的位置向量
        """
        size, dim = shape
        return sinusoidal_embeddings(ops.arange(size, dtype=keras.mixed_precision.dtype_policy().name), dim)
def align(tensor, axes, ndim=None):
    """重新对齐tensor（批量版expand_dims）
    axes：原来的第i维对齐新tensor的第axes[i]维；
    ndim：新tensor的维度。
    """
    assert len(axes) == ops.ndim(tensor)
    assert ndim or min(axes) >= 0
    ndim = ndim or max(axes) + 1
    indices = [None] * ndim
    for i in axes:
        indices[i] = slice(None)
    if keras.__version__>'3.0' and os.environ["KERAS_BACKEND"] != "jax":
        return tensor[indices]
    return tensor[tuple(indices)]
def _apply_rotary_pos_emb( tensor, cos_emb, sin_emb):
        
    x1, x2 = ops.split(tensor, 2, axis=-1)
    half_rot_tensor = ops.stack((-x2, x1), axis=-2)
    
    half_rot_tensor = ops.reshape(half_rot_tensor, ops.shape(tensor))
    
    return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

def apply_rotary_position_embeddings(sinusoidal, *tensors):
    """应用RoPE到tensors中
    其中，sinusoidal.shape=[b, n, d]，tensors为tensor的列表，而
    tensor.shape=[b, n, ..., d]。
    """
    assert len(tensors) > 0, 'at least one input tensor'
    assert all([
        list(int_shape(tensor) == int_shape(tensors[0])) for tensor in tensors[1:]
    ]), 'all tensors must have the same shape'
    ndim = ops.ndim(tensors[0])
    sinusoidal = align(sinusoidal, [0, 1, -1], ndim)
    cos_pos = ops.repeat(sinusoidal[..., 1::2], 2, -1)
    sin_pos = ops.repeat(sinusoidal[..., ::2], 2, -1)
    outputs = []
    for tensor in tensors:
        tensor2 = ops.stack([-tensor[..., 1::2], tensor[..., ::2]], ndim)
        tensor2 = ops.reshape(tensor2, ops.shape(tensor))
        outputs.append(tensor * cos_pos + tensor2 * sin_pos)
    return outputs[0] if len(outputs) == 1 else outputs
def apply_rotary_position_embeddings_keras(sinusoidal, inputs):
    ndim = ops.ndim(inputs)
    sinusoidal = align(sinusoidal, [0, 1, -1], ndim)
    dims = sinusoidal.shape[-1]
    cos_emb = sinusoidal[..., :dims//2]
    sin_emb = sinusoidal[..., dims//2:]
    inputs = ops.moveaxis(
            inputs, (-1, 1), (-1, 1)
        )
    
    output = _apply_rotary_pos_emb(inputs, cos_emb, sin_emb)
    return ops.moveaxis(
            output, (-1, 1), (-1, 1)
        )
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：
        1. y_true和y_pred的shape一致，y_true的元素是0～1
           的数，表示当前类是目标类的概率；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 和
           https://kexue.fm/archives/9064 。
    """
    y_mask = ops.not_equal(y_pred, -np.inf)
    y_neg = ops.where(y_mask, y_pred, -np.inf) + ops.log(1 - y_true)
    y_pos = ops.where(y_mask, -y_pred, -np.inf) + ops.log(y_true)
    zeros = ops.zeros_like(y_pred[..., :1])
    y_neg = ops.concatenate([y_neg, zeros], axis=-1)
    y_pos = ops.concatenate([y_pos, zeros], axis=-1)
    neg_loss = ops.logsumexp(y_neg, axis=-1)
    pos_loss = ops.logsumexp(y_pos, axis=-1)
    return neg_loss + pos_loss

def batch_gather(params, indices):
    """同tf旧版本的batch_gather
    """
    if dtype(indices)[:3] != 'int':
        indices = ops.cast(indices, 'int32')

    try:
        return ops.take(params, indices, batch_dims=ops.ndim(indices) - 1)
    except Exception as e1:
        try:
            return tf.batch_gather(params, indices)
        except Exception as e2:
            raise ValueError('%s\n%s\n' % (e1.message, e2.message))

def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    zeros = ops.zeros_like(y_pred[..., :1])
    y_pred = ops.concatenate([y_pred, zeros], axis=-1)
    if mask_zero:
        infs = zeros + np.inf
        y_pred = ops.concatenate([infs, y_pred[..., 1:]], axis=-1)
    y_pos_2 = batch_gather(y_pred, y_true)
    y_pos_1 = ops.concatenate([y_pos_2, zeros], axis=-1)
    if mask_zero:
        y_pred = ops.concatenate([-infs, y_pred[..., 1:]], axis=-1)
        y_pos_2 = batch_gather(y_pred, y_true)
    pos_loss = ops.logsumexp(-y_pos_1, axis=-1)
    all_loss = ops.logsumexp(y_pred, axis=-1)
    aux_loss = ops.logsumexp(y_pos_2, axis=-1) - all_loss
    aux_loss = ops.clip(1 - ops.exp(aux_loss), ops.epsilon(), 1)
    neg_loss = all_loss + ops.log(aux_loss)
    return pos_loss + neg_loss


def symbolic(f):
    """恒等装饰器（兼容旧版本keras用）
    """
    return f

if keras.__version__<'3.0':
    def graph_mode_decorator(f, *args, **kwargs):
        """tf 2.1与之前版本的传参方式不一样，这里做个同步
        """
        if tf.__version__ < '2.1':
            return _graph_mode_decorator(f, *args, **kwargs)
        else:
            return _graph_mode_decorator(f, args, kwargs)
    
    def recompute_grad(call):
        """重计算装饰器（用来装饰Keras层的call函数）
        关于重计算，请参考：https://arxiv.org/abs/1604.06174
        """
        if not do_recompute:
            return call
    
        def inner(self, inputs, **kwargs):
            """定义需要求梯度的函数以及重新定义求梯度过程
            （参考自官方自带的tf.recompute_grad函数）
            """
            flat_inputs = nest.flatten(inputs)
            call_args = tf_inspect.getfullargspec(call).args
            for key in ['mask', 'training']:
                if key not in call_args and key in kwargs:
                    del kwargs[key]
    
            def kernel_call():
                """定义前向计算
                """
                return call(self, inputs, **kwargs)
    
            def call_and_grad(*inputs):
                """定义前向计算和反向计算
                """
                if is_tf_keras:
                    with tape.stop_recording():
                        outputs = kernel_call()
                        outputs = tf.identity(outputs)
                else:
                    outputs = kernel_call()
    
                def grad_fn(doutputs, variables=None):
                    watches = list(inputs)
                    if variables is not None:
                        watches += list(variables)
                    with tf.GradientTape() as t:
                        t.watch(watches)
                        with tf.control_dependencies([doutputs]):
                            outputs = kernel_call()
                    grads = t.gradient(
                        outputs, watches, output_gradients=[doutputs]
                    )
                    del t
                    return grads[:len(inputs)], grads[len(inputs):]
    
                return outputs, grad_fn
    
            if is_tf_keras:  # 仅在tf >= 2.0下可用
                outputs, grad_fn = call_and_grad(*flat_inputs)
                flat_outputs = nest.flatten(outputs)
    
                def actual_grad_fn(*doutputs):
                    grads = grad_fn(*doutputs, variables=self.trainable_weights)
                    return grads[0] + grads[1]
    
                watches = flat_inputs + self.trainable_weights
                watches = [tf.convert_to_tensor(x) for x in watches]
                tape.record_operation(
                    call.__name__, flat_outputs, watches, actual_grad_fn
                )
                return outputs
            else:  # keras + tf >= 1.14 均可用
                return graph_mode_decorator(call_and_grad, *flat_inputs)
    
        return inner


ops.reshape = reshape
ops.flatten = flatten


if use_keras_2:
    
    # 给旧版keras新增symbolic（装饰器），以兼容optimizers.py
    keras.backend.symbolic = getattr(keras.backend, 'symbolic', None) or symbolic
    
    # 给tf.keras补充上logsumexp
    keras.backend.logsumexp = getattr(keras.backend, 'logsumexp', None) or tf.math.reduce_logsumexp
else:
    sys.modules['keras.ops']=ops

def slices_index(x,index,axis):
    shape = list(ops.shape(x))
    shape[axis] = index
    return ops.slice(x,ops.zeros_like(shape),shape)
custom_objects = {
    'gelu_erf': gelu_erf,
    'gelu_tanh': ops.gelu,
    'gelu': gelu_erf,
    'root_mean_square': root_mean_square,
    'swish': swish,
    'leaky_relu': leaky_relu,
    'Sinusoidal': Sinusoidal,
    'multilabel_categorical_crossentropy': multilabel_categorical_crossentropy,
    'initializer': keras.initializers.glorot_uniform,  # 就当是默认初始化方案吧
}

keras.utils.get_custom_objects().update(custom_objects)

