from bert4keras3.backend import keras, ops , np , K,recompute_grad
from keras import Layer,initializers, activations
from bert4keras3.backend import sinusoidal_embeddings,int_shape
class Embedding(keras.layers.Embedding):
    #def __init__(self, **kwargs):
    #    super(Embedding, self).__init__(dtype='float32',**kwargs)

    
    def compute_mask(self, inputs, mask=None):
        """为了适配T5，保证第一个token不被mask
        """
        #print(inputs)
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
        flag = isinstance(inputs,list)
        if self.custom_position_ids or flag :
            inputs, position_ids = inputs
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
            if self.custom_position_ids or flag :
                embeddings = ops.take(self.embeddings, position_ids,axis=0)
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
        return ops.take(self.embeddings,ops.cast(pos_ids,'int32'),axis=0)

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
        return (input_shape[0][1], input_shape[1][1], self.output_dim)

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

class RotaryEmbedding(keras.layers.Layer):
    """Rotary positional encoding layer.
    使用方法和SinusoidalPositionEmbedding一样，但是和苏神的原始rope有些变化
    """

    def __init__(
        self,
        max_wavelength=10000,
        scaling_factor=1.0,
        output_dim=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.scaling_factor = scaling_factor
        self.output_dim = output_dim
        self.built = True

    def call(self, x, start_index=0, positions=None):
        cos_emb, sin_emb = self._compute_cos_sin_embedding(
            x, start_index, positions
        )
        
        embeddings = ops.concatenate([cos_emb,sin_emb ], axis=-1)
        return embeddings

    def _compute_positions(self, inputs, start_index=0):
        seq_len = ops.shape(inputs)[1]
        positions = ops.arange(seq_len, dtype="float32")
        return positions + ops.cast(start_index, dtype="float32")

    def _compute_cos_sin_embedding(self, inputs, start_index=0, positions=None):
        feature_axis = len(inputs.shape) - 1
        sequence_axis = 1

        inverse_freq = self._get_inverse_freq(self.output_dim)

        if positions is None:
            positions = self._compute_positions(inputs, start_index)
        else:
            positions = ops.cast(positions, "float32")

        freq = ops.einsum("i,j->ij", positions, inverse_freq)
        embedding = ops.stack((freq, freq), axis=-2)
        embedding = ops.reshape(
            embedding, (*ops.shape(freq)[:-1], ops.shape(freq)[-1] * 2)
        )

        # Reshape the embedding to be broadcastable with input shape.
        if feature_axis < sequence_axis:
            embedding = ops.transpose(embedding)
        for axis in range(len(inputs.shape)):
            if axis != sequence_axis and axis != feature_axis:
                embedding = ops.expand_dims(embedding, axis)

        cos_emb = ops.cast(ops.cos(embedding), self.compute_dtype)
        sin_emb = ops.cast(ops.sin(embedding), self.compute_dtype)
        return cos_emb, sin_emb

    def _get_inverse_freq(self, rotary_dim):
        freq_range = ops.arange(0, rotary_dim, 2, dtype="float32")
        freq_range = freq_range / ops.cast(self.scaling_factor, "float32")
        inverse_freq = 1.0 / (
            self.max_wavelength
            ** (freq_range / ops.cast(rotary_dim, "float32"))
        )
        return inverse_freq

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
                "scaling_factor": self.scaling_factor,
                "output_dim":self.output_dim,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        
        return input_shape[:2] + (self.output_dim*2,)
    
class ReversibleEmbedding(keras.layers.Embedding):
    """An embedding layer which can project backwards to the input dim.

    This layer is an extension of `keras.layers.Embedding` for language models.
    This layer can be called "in reverse" with `reverse=True`, in which case the
    layer will linearly project from `output_dim` back to `input_dim`.

    By default, the reverse projection will use the transpose of the
    `embeddings` weights to project to `input_dim` (weights are "tied"). If
    `tie_weights=False`, the model will use a separate, trainable variable for
    reverse projection.

    This layer has no bias terms.

    Args:
        input_dim: Integer. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: Integer. Dimension of the dense embedding.
        tie_weights: Boolean, whether or not the matrix for embedding and
            the matrix for the `reverse` projection should share the same
            weights.
        embeddings_initializer: Initializer for the `embeddings`
            matrix (see `keras.initializers`).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix (see `keras.regularizers`).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix (see `keras.constraints`).
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out.
        reverse_dtype: The dtype for the reverse projection computation.
            Defaults to the `compute_dtype` of the layer.
        **kwargs: other keyword arguments passed to `keras.layers.Embedding`,
            including `name`, `trainable`, `dtype` etc.

    Call arguments:
        inputs: The tensor inputs to the layer.
        reverse: Boolean. If `True` the layer will perform a linear projection
            from `output_dim` to `input_dim`, instead of a normal embedding
            call. Default to `False`.

    Example:
    ```python
    batch_size = 16
    vocab_size = 100
    hidden_dim = 32
    seq_length = 50

    # Generate random inputs.
    token_ids = np.random.randint(vocab_size, size=(batch_size, seq_length))

    embedding = keras_nlp.layers.ReversibleEmbedding(vocab_size, hidden_dim)
    # Embed tokens to shape `(batch_size, seq_length, hidden_dim)`.
    hidden_states = embedding(token_ids)
    # Project hidden states to shape `(batch_size, seq_length, vocab_size)`.
    logits = embedding(hidden_states, reverse=True)
    ```

    References:
    - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    - [Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        tie_weights=True,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        reverse_dtype=None,
        **kwargs,
    ):
        super().__init__(
            input_dim,
            output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            **kwargs,
        )
        self.tie_weights = tie_weights
        self.reverse_dtype = reverse_dtype

    def build(self, inputs_shape=None):
        super().build(inputs_shape)

        if not self.tie_weights:
            self.reverse_embeddings = self.add_weight(
                name="reverse_embeddings",
                shape=(self.output_dim, self.input_dim),
                initializer=self.embeddings_initializer,
                dtype=self.dtype,
            )

    def call(self, inputs, reverse=False):
        if reverse:
            if self.tie_weights:
                kernel = ops.transpose(ops.convert_to_tensor(self.embeddings))
            else:
                kernel = self.reverse_embeddings
            if self.reverse_dtype is not None:
                inputs = ops.cast(inputs, self.reverse_dtype)
                kernel = ops.cast(kernel, self.reverse_dtype)
            return ops.matmul(inputs, kernel)

        return super().call(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tie_weights": self.tie_weights,
                "reverse_dtype": self.reverse_dtype,
            }
        )
        return config

    def save_own_variables(self, store):
        if not self.built:
            return
        super().save_own_variables(store)
        # Before Keras 3.2, the reverse weight is saved in the super() call.
        # After Keras 3.2, the reverse weight must be saved manually.
        if len(store.keys()) < len(self.weights):
            # Store the reverse embedding as the last weight.
            store[str(len(store.keys()))] = self.reverse_embeddings

    def load_own_variables(self, store):
        if not self.built:
            self.build()
        super().load_own_variables(store)
        if not self.tie_weights:
            # Last weight in the store is the reverse embedding weights.
            key = str(len(store.keys()) - 1)
            self.reverse_embeddings.assign(store[key])
    def compute_mask(self, inputs, mask=None):
        return mask
    def compute_output_spec(self, inputs, reverse=False):
        output_shape = list(inputs.shape)
        if reverse:
            output_shape[-1] = self.input_dim
        else:
            output_shape += [self.output_dim]
        return keras.KerasTensor(output_shape, dtype=self.dtype)