from bert4keras3.backend import keras, ops , np , K,recompute_grad
from keras import Layer,initializers, activations
from keras.layers import Dense
from bert4keras3.Layers_add.Embeddings import SinusoidalPositionEmbedding
from bert4keras3.backend import apply_rotary_position_embeddings,sequence_masking
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

