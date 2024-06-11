from bert4keras3.backend import keras, ops , np , K,recompute_grad,integerize_shape
from keras import Layer,initializers, activations
from keras.layers import Dense ,Dropout
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


class GemmaFeedForward(FeedForward):
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        
        self.i0_dense=keras.layers.Dense(
            self.units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
            )
        
        self.i1_dense=keras.layers.Dense(
            self.units,
            use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
                )

        self.o_dense = keras.layers.Dense(
            output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
       
    @recompute_grad
    def call(self, inputs):
        x = ops.gelu(self.i0_dense(inputs), approximate=True)* self.i1_dense(inputs)
        x = self.o_dense(x)
        return x
    
class LLamaFeedForward(FeedForward):
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        self._feedforward_gate_dense = keras.layers.Dense(
            self.units,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            name="feedforward_gate_dense",
        )
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.units,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            name="feedforward_intermediate_dense",
        )
        

        self._feedforward_output_dense = keras.layers.Dense(
            output_dim,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            name="feedforward_output_dense",
        )
    @recompute_grad
    def call(self, x):

        activation = activations.get(self.activation[0])
        gate_output = self._feedforward_gate_dense(x)
        gate_output = activation(gate_output)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(ops.multiply(x, gate_output))
        return x#

