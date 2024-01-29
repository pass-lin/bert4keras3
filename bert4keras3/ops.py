# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:09:16 2023

@author: Administrator
"""
import tensorflow as tf
from distutils.util import strtobool
import os, sys
is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))
import numpy as np
if is_tf_keras:
    sys.modules['keras'] = tf.keras
import keras
import keras.backend as K

#numpy like api
absolute=K.abs

add=tf.add

all=K.all

amax=K.max

amin=K.min

any=K.any

def append(x1, x2, axis=0):
    if axis==None:
        raise('must specified axis')
    return K.concatenate([x1,x2],axis)
    
arange=K.arange
def arccos(x):
    return tf.math.acos(x)

def arccosh(x):
    return tf.math.acosh(x)

def arcsin(x):
    return tf.math.asin(x)

def arcsinh(x):
    return tf.math.asinh(x)

def arctan(x):
    return tf.math.atanh(x)

def arctan2(x):
    return tf.math.atanh2(x)

def arctanh(x):
    return tf.math.atanh(x)

argmax=K.argmax

argmin=K.argmin

def argsort(x,axis=-1):
    return tf.argsort(x,axis)

def array(x, dtype=None):
    return K.constant(x, dtype=None)

def average(x, axis=None, weights=None):
    if weights!=None:
        x*=weights
    return tf.reduce_mean(x,axis)

bincount=tf.math.bincount

broadcast_to=tf.broadcast_to

ceil=tf.math.ceil
clip=K.clip

concatenate=K.concatenate

conj=conjugate=tf.math.conj


cos=tf.math.cos

cosh=tf.math.cosh

def count_nonzero(x, axis=None):
    return tf.math.count_nonzero(x, axis)

cumprod=K.cumprod

ndim=K.ndim

def diag(x, k=0):
    if ndim(x)==1:
        return tf.linalg.diag(x,k=k)
    return tf.linalg.diag_part(x,k=k)
    
divide=tf.divide

dot=K.dot

einsum=tf.einsum

equal=K.equal

exp=K.exp

expand_dims=K.expand_dims

def expm1(x):
    return exp(x)-1

eye=K.eye

flip=K.reverse

floor=tf.floor

def floor_divide(x1, x2):
    return floor(x1/x2)

def full(shape, fill_value, dtype=None):
    return K.zeros(shape,dtype)+fill_value

def full_like(x, fill_value, dtype=None):
    return K.zeros_like(x,dtype)+fill_value

greater=K.greater

greater_equal=K.greater_equal

def hstack(xs):
    return concatenate(xs,axis=1)

def identity(n, dtype=None):
    return K.eye(n,dtype)

imag=tf.math.imag

def isclose(x1, x2):
    return tf.debugging.assert_near(x,y)

isfinite=tf.math.is_finite

isinf=tf.math.is_inf

isnan=tf.math.is_nan

less=K.less

less_equal=K.less_equal

log=K.log

def log1p(x):
    return log(1+x)

def log2(x):
    return log(x)/log(2)

def log10(x):
    return log(x)/log(10)

def logaddexp(x1, x2):
    return log(exp(x1) + exp(x2))

logical_and=tf.logical_and

logical_not=tf.logical_not

logical_or=tf.logical_or



def matmul(x1,x2):
    return tf.matmul(x1,x2)

def max(x, axis=None, keepdims=False, initial=None):
    x=K.max(x,axis,keepdims)
    if initial!=None:
        x=K.maximum(x,initial)
    return x

maximum=K.maximum

minimum=K.minimum
        
mean=K.mean

meshgrid=tf.meshgrid

def min(x, axis=None, keepdims=False, initial=None):
    x=K.min(x,axis,keepdims)
    if initial!=None:
        x=K.minimum(x,initial)
    return x

mod=tf.math.mod

multiply=tf.multiply

negative=tf.negative

not_equal=K.not_equal

ones=K.ones

ones_like=K.ones_like

pad=tf.pad

power=K.pow

def prod(x, axis=None, keepdims=False, dtype=None):
    x=K.prod(x,axis,keepdims)
    if dtype!=None:
        x=K.cast(x,dtype)
    return x

def ravel(x):
    return K.reshape(x,[-1])

real=tf.math.real

reciprocal=tf.math.reciprocal

repeat=tf.repeat

reshape=K.reshape

roll=tf.roll

sign=K.sign

sin=tf.math.sin

sinh=tf.math.sinh

size=tf.size

def sort(x, axis=-1):
    return sort(x,axis)



sqrt=tf.math.sqrt

square=tf.math.square

squeeze=K.squeeze

stack=K.stack

std=K.std

subtract=tf.subtract

sum=K.sum

def swapaxes(x, axis1, axis2):
    perm=[i for i in range(ndim(x))]
    perm[axis1]=axis2
    perm[axis2]=axis1
    return tf.transpose(x,perm)

def take(x, indices, axis=None):
    return tf.gather(x,indices,axis=axis)

tan=tf.math.tan

tanh=tf.math.tanh

tensordot=tf.tensordot

tile=K.tile

def transpose(a,axes=None):
    
    if axes==None:
        return K.transpose(a) 
    return tf.transpose(a,axes)


def tri(N, M=None, k=0, dtype='float32'):
    if M==None:
        M=N
    t1 = K.arange(0, M)
    t2 = K.arange(0, N)
    if k!=0:
        t2+=k
    mask = t1[None, :] <= t2[:, None]
    return K.cast(mask,dtype)
        
def tril(x, k=0):
    shape=K.shape(x)
    N,M=shape[-2],shape[-1]
    t1 = K.arange(0, M)
    t2 = K.arange(0, N)
    if k!=0:
        t2+=k
    mask = t1[None, :] <= t2[:, None]
    return tf.where(mask,x,tf.zeros(shape,x.dtype))

def triu(x, k=0):
    shape=K.shape(x)
    N,M=shape[-2],shape[-1]
    t1 = K.arange(0, M)
    t2 = K.arange(0, N)
    if k!=0:
        t2+=k
    mask = t1[None, :] >= t2[:, None]
    return tf.where(mask,x,tf.zeros(shape,x.dtype))

true_divide=divide

var=K.var

where=tf.where

zeros=tf.zeros

zeros_like=K.zeros_like

#NN ops
batch_normalization=K.batch_normalization

binary_crossentropy=K.binary_crossentropy

categorical_crossentropy=K.categorical_crossentropy

def _convert_data_format(data_format, ndim):
    if data_format == "channels_last":
        if ndim == 3:
            return "NWC"
        elif ndim == 4:
            return "NHWC"
        elif ndim == 5:
            return "NDHWC"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    elif data_format == "channels_first":
        if ndim == 3:
            return "NCW"
        elif ndim == 4:
            return "NCHW"
        elif ndim == 5:
            return "NCDHW"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    else:
        raise ValueError(
            f"Invalid data_format: {data_format}. "
            'Expected values are ["channels_first", "channels_last"]'
        )


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    def _conv():
        tf_data_format = _convert_data_format(data_format, len(inputs.shape))
        return tf.nn.convolution(
            inputs,
            kernel,
            strides,
            padding.upper(),
            data_format=tf_data_format,
            dilations=dilation_rate,
        )

    data_format = standardize_data_format(data_format)
    if data_format == "channels_last":
        channels = inputs.shape[-1]
    else:
        channels = inputs.shape[1]
    if channels != kernel.shape[-2]:
        # If kernel's in_channel does not match input's channels,  it indicates
        # convolution is broken down into groups.
        return _conv()
    return _conv()

def _get_output_shape_given_tf_padding(
    input_size, kernel_size, strides, padding, output_padding, dilation_rate
):
    if input_size is None:
        return None

    assert padding.lower() in {"valid", "same"}

    kernel_size = (kernel_size - 1) * dilation_rate + 1

    if padding.lower() == "valid":
        output_padding = (
            max(kernel_size, strides) - kernel_size
            if output_padding is None
            else output_padding
        )
        return (input_size - 1) * strides + kernel_size + output_padding

    else:
        if output_padding is None:
            return input_size * strides
        else:
            return (input_size - 1) * strides + kernel_size % 2 + output_padding
        
        
def compute_conv_transpose_output_shape(
    input_shape,
    kernel_size,
    filters,
    strides,
    padding,
    output_padding=None,
    data_format="channels_last",
    dilation_rate=1,
):
    num_spatial_dims = len(input_shape) - 2
    kernel_spatial_shape = kernel_size

    if isinstance(output_padding, int):
        output_padding = (output_padding,) * len(kernel_spatial_shape)
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims

    if data_format == "channels_last":
        input_spatial_shape = input_shape[1:-1]
    else:
        input_spatial_shape = input_shape[2:]

    output_shape = []
    for i in range(num_spatial_dims):
        current_output_padding = (
            None if output_padding is None else output_padding[i]
        )

        shape_i = _get_output_shape_given_tf_padding(
            input_size=input_spatial_shape[i],
            kernel_size=kernel_spatial_shape[i],
            strides=strides[i],
            padding=padding,
            output_padding=current_output_padding,
            dilation_rate=dilation_rate[i],
        )
        output_shape.append(shape_i)

    if data_format == "channels_last":
        output_shape = [input_shape[0]] + output_shape + [filters]
    else:
        output_shape = [input_shape[0], filters] + output_shape
    return output_shape

def standardize_data_format(data_format):
    if data_format is None:
        return "channels_last"
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format

def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1,
):
    data_format = standardize_data_format(data_format)
    tf_data_format = _convert_data_format(data_format, len(inputs.shape))
    kernel_size = kernel.shape[:-2]
    filters = kernel.shape[-2]
    input_shape = list(inputs.shape)
    symbolic_shape = tf.shape(inputs)
    for i, e in enumerate(input_shape):
        if e is None:
            input_shape[i] = symbolic_shape[i]
    output_shape = compute_conv_transpose_output_shape(
        input_shape,
        kernel_size,
        filters,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    )

    return tf.nn.conv_transpose(
        inputs,
        kernel,
        output_shape,
        strides,
        padding=padding.upper(),
        data_format=tf_data_format,
        dilations=dilation_rate,
    )

def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = standardize_data_format(data_format)
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(
            "`inputs` rank must be 3 (1D conv) or 4 (2D conv). Received: "
            "{inputs.ndim}."
        )
    # Because we use `tf.nn.depthwise_conv2d` for both 1D and 2D convs, we set
    # `tf_data_format` using 2D conv format.
    tf_data_format = _convert_data_format(data_format, 4)
    padding = padding.upper()
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    if num_spatial_dims == 1:
        # 1D depthwise conv.
        if data_format == "channels_last":
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        kernel = tf.expand_dims(kernel, axis=0)

        dilation_rate = None if dilation_rate is None else (1,) + dilation_rate

        outputs = tf.nn.depthwise_conv2d(
            inputs,
            kernel,
            strides,
            padding,
            data_format=tf_data_format,
            dilations=dilation_rate,
        )
        return tf.squeeze(outputs, [spatial_start_dim])

    if data_format == "channels_last":
        strides = (1,) + strides + (1,)
        spatial_start_dim = 1
    else:
        strides = (1, 1) + strides
        spatial_start_dim = 2
    return tf.nn.depthwise_conv2d(
        inputs,
        kernel,
        strides,
        padding,
        data_format=tf_data_format,
        dilations=dilation_rate,
    )

elu=K.elu

def gelu(x):
    """基于Tanh近似计算的gelu函数
    """
    cdf = 0.5 * (
        1.0 + K.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3))))
    )
    return x * cdf
def hard_sigmoid(x):
    x = x / 6.0 + 0.5
    return K.clip(x, 0.0, 1.0)

def leaky_relu(x, negative_slope=0.2):
    return tf.nn.leaky_relu(x, alpha=negative_slope)

def log_sigmoid(x):
    return tf.math.log_sigmoid(x)

def log_softmax(x, axis=-1):
    if axis is None:
        # Unlike numpy, tf will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        output = tf.reshape(x, [-1])
        output = tf.nn.log_softmax(output, axis=-1)
        return tf.reshape(output, tf.shape(x))
    return tf.nn.log_softmax(x, axis=axis)

def _transpose_spatial_inputs(inputs):
    num_spatial_dims = len(inputs.shape) - 2
    # Tensorflow pooling does not support `channels_first` format, so
    # we need to transpose to `channels_last` format.
    if num_spatial_dims == 1:
        inputs = tf.transpose(inputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        inputs = tf.transpose(inputs, (0, 2, 3, 1))
    elif num_spatial_dims == 3:
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))
    else:
        raise ValueError(
            "Pooling inputs's shape must be 3, 4 or 5, corresponding to 1D, 2D "
            f"and 3D inputs. But received shape: {inputs.shape}."
        )
    return inputs


def _transpose_spatial_outputs(outputs):
    # Undo the tranpose in `_transpose_spatial_inputs`.
    num_spatial_dims = len(outputs.shape) - 2
    if num_spatial_dims == 1:
        outputs = tf.transpose(outputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        outputs = tf.transpose(outputs, (0, 3, 1, 2))
    elif num_spatial_dims == 3:
        outputs = tf.transpose(outputs, (0, 4, 1, 2, 3))
    return outputs


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = standardize_data_format(data_format)
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    tf_data_format = _convert_data_format("channels_last", len(inputs.shape))
    if data_format == "channels_first":
        # Tensorflow pooling does not support `channels_first` format, so
        # we need to transpose to `channels_last` format.
        inputs = _transpose_spatial_inputs(inputs)

    outputs = tf.nn.avg_pool(
        inputs,
        pool_size,
        strides,
        padding,
        tf_data_format,
    )
    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs

def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = standardize_data_format(data_format)
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    tf_data_format = _convert_data_format("channels_last", len(inputs.shape))
    if data_format == "channels_first":
        # Tensorflow pooling does not support `channels_first` format, so
        # we need to transpose to `channels_last` format.
        inputs = _transpose_spatial_inputs(inputs)

    outputs = tf.nn.max_pool(
        inputs,
        pool_size,
        strides,
        padding,
        tf_data_format,
    )
    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs

def one_hot(x, num_classes, axis=-1, dtype="float32"):
    return tf.one_hot(x, num_classes, axis=axis, dtype=dtype)


def moments(x, axes, keepdims=False):
    need_cast = False
    ori_dtype = x.dtype
    if ori_dtype == "float16":
        need_cast = True
        x = cast(x, "float32")

    mean = tf.reduce_mean(x, axes, keepdims=True)

    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    # Note: stop_gradient does not change the gradient to the mean, because that
    # gradient is zero.
    # The substraction operation does not guarantee a non-negative
    # result given float precision, so we clamp it to 0.
    variance = tf.maximum(
        tf.reduce_mean(tf.square(x), axis=axes, keepdims=True)
        - K.square(K.stop_gradient(mean)),
        0.0,
    )

    if not keepdims:
        mean = tf.squeeze(mean, axes)
        variance = tf.squeeze(variance, axes)
    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = K.clip(mean, tf.float16.min, tf.float16.max)
        variance = K.clip(variance, tf.float16.min, tf.float16.max)
        mean = K.cast(mean, ori_dtype)
        variance = K.cast(variance, ori_dtype)
    return mean, variance

def multi_hot(x, num_classes, axis=-1, dtype="float32"):
    reduction_axis = 1 if len(x.shape) > 1 else 0
    outputs = tf.reduce_max(
        one_hot(cast(x, "int32"), num_classes, axis=axis, dtype=dtype),
        axis=reduction_axis,
    )
    return outputs

def relu(x):
    return tf.nn.relu(x)


def relu6(x):
    return tf.nn.relu6(x)

sigmoid=tf.nn.sigmoid

def tanh(x):
    return tf.nn.tanh(x)

def softplus(x):
    return tf.math.softplus(x)

def softsign(x):
    return tf.nn.softsign(x)


silu=tf.nn.swish


def log_sigmoid(x):
    return tf.math.log_sigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    return tf.nn.leaky_relu(x, alpha=negative_slope)


def hard_sigmoid(x):
    x = x / 6.0 + 0.5
    return tf.clip_by_value(x, 0.0, 1.0)


elu=K.elu


def selu(x):
    return tf.nn.selu(x)





def softmax(x, axis=-1):
    logits = x
    if axis is None:
        # Unlike numpy, tf will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        output = tf.reshape(x, [-1])
        output = tf.nn.softmax(output, axis=-1)
        output = tf.reshape(output, tf.shape(x))
    else:
        output = tf.nn.softmax(x, axis=axis)
    output._keras_logits = logits
    return output

def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = standardize_data_format(data_format)
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(
            "`num_spatial_dims` must be 1 or 2. Received: "
            f"num_spatial_dims={num_spatial_dims}."
        )
    # Because we use `tf.nn.separable_conv2d` for both 1D and 2D convs, we set
    # `tf_data_format` using 2D conv format.
    tf_data_format = _convert_data_format(data_format, 4)
    padding = padding.upper()
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    if num_spatial_dims == 1:
        # 1D depthwise conv.
        if data_format == "channels_last":
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(depthwise_kernel, axis=0)
        pointwise_kernel = tf.expand_dims(pointwise_kernel, axis=0)
        dilation_rate = None if dilation_rate is None else (1,) + dilation_rate

        outputs = tf.nn.separable_conv2d(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            strides,
            padding,
            data_format=tf_data_format,
            dilations=dilation_rate,
        )
        return tf.squeeze(outputs, [spatial_start_dim])

    if data_format == "channels_last":
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides
    return tf.nn.separable_conv2d(
        inputs,
        depthwise_kernel,
        pointwise_kernel,
        strides,
        padding,
        data_format=tf_data_format,
        dilations=dilation_rate,
    )

sparse_categorical_crossentropy=K.sparse_categorical_crossentropy

cast=K.cast

cond=tf.cond

def slice_update(inputs, start_indices, updates):
    return tf.compiler.tf2xla.python.xla.dynamic_update_slice(inputs, updates, start_indices)

def convert_to_numpy(x):
    if isinstance(x, tf.SparseTensor):
        x = tf.sparse.to_dense(x)
    elif isinstance(x, tf.IndexedSlices):
        x = tf.convert_to_tensor(x)
    return np.array(x)

convert_to_tensor=tf.convert_to_tensor

erf=tf.math.erf

def extract_sequences(x, sequence_length, sequence_stride):
    return tf.signal.frame(
        x,
        frame_length=sequence_length,
        frame_step=sequence_stride,
        axis=-1,
        pad_end=False,
    )

def fori_loop(lower, upper, body_fun, init_val):
    return tf.while_loop(
        lambda i, val: i < upper,
        lambda i, val: (i + 1, body_fun(i, val)),
        (lower, init_val),
    )[1]


is_tensor=tf.is_tensor

def logsumexp(x, axis=None, keepdims=False):
    return tf.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims)

def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    if mode == "reduced":
        return tf.linalg.qr(x)
    return tf.linalg.qr(x, full_matrices=True)

def rsqrt(x):
    return tf.math.rsqrt(x)

def scatter(indices, values, shape):
    return tf.scatter_nd(indices, values, shape)

def scatter_update(inputs, indices, updates):
    return tf.tensor_scatter_nd_update(inputs, indices, updates)

def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        return tf.math.segment_sum(data, segment_ids)
    else:
        if num_segments is None:
            unique_segment_ids, _ = tf.unique(segment_ids)
            num_segments = tf.shape(unique_segment_ids)[0]
        return tf.math.unsorted_segment_sum(data, segment_ids, num_segments)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        return tf.math.segment_max(data, segment_ids)
    else:
        if num_segments is None:
            unique_segment_ids, _ = tf.unique(segment_ids)
            num_segments = tf.shape(unique_segment_ids)[0]
        return tf.math.unsorted_segment_max(data, segment_ids, num_segments)


def top_k(x, k, sorted=True):
    return tf.math.top_k(x, k, sorted=sorted)


def in_top_k(targets, predictions, k):
    return tf.math.in_top_k(targets, predictions, k)

shape=K.shape

slice=tf.slice

def solve(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return tf.linalg.solve(a, b)

stop_gradient=K.stop_gradient

def unstack(x, num=None, axis=0):
    return tf.unstack(x, num=num, axis=axis)

while_loop=tf.while_loop

def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = tf.signal.fft(complex_input)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = tf.signal.fft2d(complex_input)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def rfft(x, fft_length=None):
    if fft_length is not None:
        fft_length = [fft_length]
    complex_output = tf.signal.rfft(x, fft_length=fft_length)
    return tf.math.real(complex_output), tf.math.imag(complex_output)

def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    # `convert_to_tensor` does not support passing complex tensors. We separate
    # the input out into real and imaginary and convert them separately.
    real, imag = x
    real = convert_to_tensor(real)
    imag = convert_to_tensor(imag)
    # Check shapes.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not real.dtype.is_floating or not imag.dtype.is_floating:
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    complex_input = tf.dtypes.complex(real, imag)
    return complex_input
def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    if fft_length is not None:
        fft_length = [fft_length]
    return tf.signal.irfft(complex_input, fft_length)


def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    if standardize_dtype(x.dtype) not in {"float32", "float64"}:
        raise TypeError(
            "Invalid input type. Expected `float32` or `float64`. "
            f"Received: input type={x.dtype}"
        )
    if fft_length < sequence_length:
        raise ValueError(
            "`fft_length` must equal or larger than `sequence_length`. "
            f"Received: sequence_length={sequence_length}, "
            f"fft_length={fft_length}"
        )
    if isinstance(window, str):
        if window not in {"hann", "hamming"}:
            raise ValueError(
                "If a string is passed to `window`, it must be one of "
                f'`"hann"`, `"hamming"`. Received: window={window}'
            )
    x = convert_to_tensor(x)

    if center:
        pad_width = [(0, 0) for _ in range(len(x.shape))]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        x = tf.pad(x, pad_width, mode="reflect")

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win_array = tf.signal.hann_window(
                    sequence_length, periodic=True, dtype=x.dtype
                )
            else:
                win_array = tf.signal.hamming_window(
                    sequence_length, periodic=True, dtype=x.dtype
                )
        else:
            win_array = convert_to_tensor(window, dtype=x.dtype)
        if len(win_array.shape) != 1 or win_array.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win_array.shape}"
            )
        win_array = tf.pad(win_array, [[l_pad, r_pad]])

        def win(frame_step, dtype):
            return win_array

    else:
        win = None

    result = tf.signal.stft(
        x,
        frame_length=(sequence_length + l_pad + r_pad),
        frame_step=sequence_stride,
        fft_length=fft_length,
        window_fn=win,
    )
    return tf.math.real(result), tf.math.imag(result)


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    complex_input = _get_complex_tensor_from_tuple(x)
    dtype = tf.math.real(complex_input).dtype

    expected_output_len = fft_length + sequence_stride * (
        tf.shape(complex_input)[-2] - 1
    )
    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win_array = tf.signal.hann_window(
                    sequence_length, periodic=True, dtype=dtype
                )
            else:
                win_array = tf.signal.hamming_window(
                    sequence_length, periodic=True, dtype=dtype
                )
        else:
            win_array = convert_to_tensor(window, dtype=dtype)
        if len(win_array.shape) != 1 or win_array.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win_array.shape}"
            )
        win_array = tf.pad(win_array, [[l_pad, r_pad]])
        win = tf.signal.inverse_stft_window_fn(
            sequence_stride, lambda frame_step, dtype: win_array
        )
    else:
        win = None

    x = tf.signal.inverse_stft(
        complex_input,
        frame_length=(sequence_length + l_pad + r_pad),
        frame_step=sequence_stride,
        fft_length=fft_length,
        window_fn=win,
    )

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center is True:
        end = -(fft_length // 2)
    else:
        end = expected_output_len
    return x[..., start:end]