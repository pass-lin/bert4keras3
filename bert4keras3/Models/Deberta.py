from bert4keras3.Models.Roformers import *
from bert4keras3.Layers_add.Embeddings import DebertaRelativeEmbedding
from bert4keras3.Layers_add.Attentions import DisentangledSelfAttention
class MaskedLMHead(keras.layers.Layer):
    """Masked Language Model (MaskedLM) head.

    This layer takes two inputs:

     - `inputs`: which should be a tensor of encoded tokens with shape
       `(batch_size, sequence_length, hidden_dim)`.
     - `mask_positions`: which should be a tensor of integer positions to
       predict with shape `(batch_size, masks_per_sequence)`.

    The token encodings should usually be the last output of an encoder model,
    and mask positions should be the integer positions you would like to
    predict for the MaskedLM task.

    The layer will first gather the token encodings at the mask positions. These
    gathered tokens will be passed through a dense layer the same size as
    encoding dimension, then transformed to predictions the same size as the
    input vocabulary. This layer will produce a single output with shape
    `(batch_size, masks_per_sequence, vocabulary_size)`, which can be used to
    compute an MaskedLM loss function.

    This layer is often be paired with `keras_nlp.layers.MaskedLMMaskGenerator`,
    which will help prepare inputs for the MaskedLM task.

    Args:
        vocabulary_size: The total size of the vocabulary for predictions.
        token_embedding: Optional. A `keras_nlp.layers.ReversibleEmbedding`
            instance. If passed, the layer will be used to project from the
            `hidden_dim` of the model to the output `vocabulary_size`.
        intermediate_activation: The activation function of intermediate dense layer.
        activation: The activation function for the outputs of the layer.
            Usually either `None` (return logits), or `"softmax"`
            (return probabilities).
        layer_norm_epsilon: float. The epsilon value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and multiheaded
            attention layers. Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and multiheaded
            attention layers. Defaults to `"zeros"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Example:

    ```python
    batch_size = 16
    vocab_size = 100
    hidden_dim = 32
    seq_length = 50

    # Generate random inputs.
    token_ids = np.random.randint(vocab_size, size=(batch_size, seq_length))
    # Choose random positions as the masked inputs.
    mask_positions = np.random.randint(seq_length, size=(batch_size, 5))

    # Embed tokens in a `hidden_dim` feature space.
    token_embedding = keras_nlp.layers.ReversibleEmbedding(
        vocab_size,
        hidden_dim,
    )
    hidden_states = token_embedding(token_ids)

    preds = keras_nlp.layers.MaskedLMHead(
        vocabulary_size=vocab_size,
        token_embedding=token_embedding,
        activation="softmax",
    )(hidden_states, mask_positions)
    ```

    References:
     - [Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)
    """

    def __init__(
        self,
        vocabulary_size=None,
        token_embedding=None,
        intermediate_activation="relu",
        activation=None,
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs, autocast=False)

        self.vocabulary_size = vocabulary_size
        self.token_embedding = token_embedding
        self.intermediate_activation = keras.activations.get(
            intermediate_activation
        )
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        if vocabulary_size is None and token_embedding is None:
            raise ValueError(
                "One of `vocabulary_size` or `token_embedding` must be set. "
                "Received: `vocabulary_size=None`, `token_embedding=None`"
            )

        if token_embedding:
            if vocabulary_size and vocabulary_size != token_embedding.input_dim:
                raise ValueError(
                    "`vocabulary_size` should match the input dimension of the "
                    "of `token_embedding`. Received: "
                    f"`vocabulary_size={vocabulary_size}`, "
                    f"`token_embedding.input_dim={token_embedding.input_dim}`"
                )
            self.vocabulary_size = token_embedding.input_dim

    def build(self, inputs_shape, mask_positions_shape=None):
        if self.token_embedding is not None:
            feature_size = self.token_embedding.output_dim
        else:
            feature_size = inputs_shape[-1]

        self._intermediate_dense = keras.layers.Dense(
            feature_size,
            activation=self.intermediate_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
            name="intermediate_dense",
        )
        self._intermediate_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="intermediate_layer_norm",
        )
        # The gather length does not affect any of our built variables, so
        # we can pass any value here.
        gather_length = None
        shape = (inputs_shape[0], gather_length, inputs_shape[-1])
        self._intermediate_dense.build(shape)
        shape = (inputs_shape[0], gather_length, feature_size)
        self._intermediate_layer_norm.build(shape)
        if self.token_embedding is None:
            self._kernel = self.add_weight(
                name="output_kernel",
                shape=[feature_size, self.vocabulary_size],
                initializer=self.kernel_initializer,
                dtype=self.dtype,
            )
        self._bias = self.add_weight(
            name="output_bias",
            shape=[self.vocabulary_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
        )
        self.built = True

    def call(self,inputs, mask_positions ):
        if keras.config.backend() == "tensorflow":
            import tensorflow as tf

            # On the tf backend, we need to work around an issue with dynamic
            # shape broadcasting in take_along_axis.
            x = tf.gather(inputs, mask_positions, batch_dims=1)
        else:
            # Gather the encoded tokens at the masked indices.
            mask_positions = ops.expand_dims(mask_positions, axis=-1)
            x = ops.take_along_axis(inputs, mask_positions, axis=1)

        # Apply a trainable linear transformation and a layer norm.
        x = self._intermediate_dense(x)
        x = self._intermediate_layer_norm(x)

        # Transform encodings to vocabulary_size predictions.
        if self.token_embedding:
            outputs = self.token_embedding(x, mode='dense')
        else:
            outputs = ops.matmul(x, self._kernel)
        outputs = ops.cast(outputs, self.compute_dtype)
        outputs = outputs + self._bias

        # Apply a final activation.
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    @classmethod
    def from_config(cls, config):
        embedding = config.get("token_embedding")
        if embedding:
            config["token_embedding"] = keras.layers.deserialize(embedding)
        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        embedding_config = None
        if self.token_embedding:
            embedding_config = keras.layers.serialize(self.token_embedding)
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "token_embedding": embedding_config,
                "intermediate_activation": keras.activations.serialize(
                    self.intermediate_activation
                ),
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config

    def compute_output_shape(self, inputs_shape, mask_positions_shape):
        return mask_positions_shape + (self.vocabulary_size,)
class Deberta(RoFormerV2):
    def __init__(self,bucket_size,**kwargs):
        super().__init__(**kwargs)
        self.bucket_size = bucket_size
        if self.hidden_act=='gelu':
            self.hidden_act = keras.activations.gelu
        #如果你的最大长度小于自定义长度，应该考虑在输入的时候更改你的最大长度,比如输入参数max_position=2048
        assert self.sequence_length is None or self.sequence_length<=self.max_position
    def get_inputs(self):
        inputs = super().get_inputs()
        self.padding_mask = self.apply(
                inputs=inputs[0],
                layer=keras.layers.Lambda,
                function=lambda x:ops.cast(ops.expand_dims(ops.not_equal(x,0), axis=1),'int32'),
                name='Padding-Mask'
                )
        if self.with_mlm:
            self.mask_positions = keras.Input(
                    shape=(None,), dtype="int32", name="mask_positions"
                )
            inputs.append(self.mask_positions)
        return inputs
    def compute_position_bias(self, x):
        if self.position_bias is None:
            self.position_bias = self.apply(
                inputs=x,
                layer=DebertaRelativeEmbedding,
                hidden_dim=self.hidden_size,
                bucket_size=self.bucket_size,
                layer_norm_epsilon=1e-7,
                kernel_initializer=self.initializer,
                name='Embedding-Deberta-Position'
            )

        return self.position_bias
    def apply_main_layers(self, inputs, index):
        x = inputs
        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.padding_mask
        position_bias = self.compute_position_bias(x)
        xi = x =inputs
        x = self.apply(
            inputs=x,
            layer=DisentangledSelfAttention,
            arguments={'rel_embeddings':position_bias,
                       'attention_mask':attention_mask},
            num_heads=self.num_attention_heads,
            hidden_dim=self.hidden_size,
            max_position_embeddings=self.max_position,
            bucket_size=self.bucket_size,
            dropout=self.dropout_rate,
            kernel_initializer=self.initializer,
            bias_initializer=self.initializer,
            name=attention_name,
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=x,
            epsilon=1e-7,
            layer=keras.layers.LayerNormalization,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=keras.layers.LayerNormalization,
            epsilon=1e-7,
            name='%s-Norm' % feed_forward_name
        )

        return x
    def apply_final_layers(self, inputs):
        if self.with_mlm:
            return self.apply(
                    inputs=inputs,
                    layer=MaskedLMHead,
                    vocabulary_size=self.vocab_size,
                    token_embedding=self.layers['Embedding-Token'],
                    intermediate_activation=self.hidden_act,
                    kernel_initializer=self.initializer,   
                    arguments={'mask_positions':self.mask_positions},
                    activation = self.with_mlm if self.with_mlm!=True else 'softmax', 
                    name="MLM_Head",
                )
        else:
            return inputs
    def apply_embeddings(self,inputs):
        self.apply(
            layer=keras.layers.LayerNormalization,
            epsilon=1e-7,
            name='Embedding-Norm'
        )
        return super().apply_embeddings(inputs)