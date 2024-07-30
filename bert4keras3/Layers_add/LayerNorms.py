from bert4keras3.backend import keras, ops , np , K,recompute_grad,integerize_shape,align
from keras import Layer,initializers,activations
from keras.layers import Dense ,Dropout
from bert4keras3.backend import divide_no_nan
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


class RMSNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(input_shape[-1],),
            initializer="zeros",
        )
        self.built = True

    def call(self, x):
        # Always compute normalization in float32.
        x = ops.cast(x, "float32")
        scale = ops.cast(self.scale, "float32")
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        normed_inputs = x * ops.reciprocal(ops.sqrt(var + self.epsilon))
        normed_inputs = normed_inputs * (1 + scale)
        return ops.cast(normed_inputs, self.compute_dtype)
    def compute_mask(self, inputs, mask=None):
        return mask
class LlamaLayerNorm(keras.layers.Layer):
    """A normalization layer for Llama that implements RMS normalization."""

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[-1]
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(dim,),
            initializer="ones",
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        x = ops.cast(x, "float32")
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        
        return ops.cast(x, self.compute_dtype) * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
    def compute_mask(self, inputs, mask=None):
        return mask
class GroupNorm(Layer):
    def __init__(self,hidden_size,head_size,epsilon=64*1e-5,name="group_norm",**kwargs):
        super(GroupNorm,self).__init__(name=name,**kwargs)

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = hidden_size // head_size
        self.epsilon =epsilon
        assert hidden_size % head_size == 0

    def call(self,inputs):
        B,T,C = ops.shape(inputs)
        x = ops.reshape(inputs,(B,T,self.num_heads,self.head_size))
        x =  ops.reshape(self.scale,(1,1,self.num_heads,self.head_size)) * self.group_ln(x) +  ops.reshape(self.offset,(1,1,self.num_heads,self.head_size))
        o = ops.reshape(x,(B,T,C))
        return o
    def compute_output_shape(self, input_shape):
        return input_shape
    def build(self, input_shape):
        super().build(input_shape)
        self.scale = self.add_weight(shape=(self.num_heads,self.head_size))
        self.offset = self.add_weight(shape=(self.num_heads,self.head_size))
        self.group_ln = keras.layers.LayerNormalization(epsilon=64*1e-5,dtype=self.dtype)
    def get_config(self):
        config = {
            'head_size':self.head_size,
            'hidden_size':self.hidden_size,
            'epsilon':self.epsilon
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))