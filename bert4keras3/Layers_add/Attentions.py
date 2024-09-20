from bert4keras3.backend import keras, ops , np , K,recompute_grad,int_shape
from keras import Layer,initializers,activations
from keras.layers import Dense ,Dropout
from bert4keras3.backend import apply_rotary_position_embeddings,integerize_shape,apply_rotary_position_embeddings_keras
from bert4keras3.backend import attention_normalize,align,enable_flashatt
from bert4keras3.Layers_add.LayerNorms import ScaleOffset
from bert4keras3.Layers_add.Embeddings import RotaryEmbedding
if enable_flashatt:
    from bert4keras3.backend import flash_mha
import math
def repeat_kv(x,num):
    #x shape is [batch_size,seq_len,heads_heads_dims]
    shape = int_shape(x)
    shape[2]*=num
    x = ops.repeat(x[:, :, None, :, :],num,axis=2)#[batch_size,seq_len,num,heads_heads_dims]
    return ops.reshape(x,list(shape))
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
        o_bias=None,
        query_head=None,
        use_EinsumDense = False,
        rope_mode='su',#su代表苏神实现，keras代表keras_nlp实现
        max_wavelength=10_000.0,
        scaling_factor=1.0,
        flatten_o_dense =  True,#为了适配gemma
        GQA_mode=False,
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.GQA_mode = GQA_mode
        self.head_size = head_size
        self.flatten_o_dense=flatten_o_dense
        self.max_wavelength = max_wavelength
        self.scaling_factor = scaling_factor
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.o_bias = o_bias
        self.use_EinsumDense = use_EinsumDense
        self.query_head = query_head
        self.rope_mode=rope_mode
        if self.o_bias==None:
            self.o_bias = use_bias
        if self.query_head==None:
            self.query_head = heads
        if self.query_head<heads:
            raise('query_head should higher than num heads')
        elif self.query_head%heads!=0:
            raise('query_head a should be divisible by heads')
    def build(self, input_shape):
        if self.use_EinsumDense:
            self.query_dense = keras.layers.EinsumDense(
            "btd,ndh->btnh",
            output_shape=(None, self.query_head, self.key_size),
            kernel_initializer=self.kernel_initializer,
            name="query",
            )
            self.query_dense.build(input_shape[0])

            self.key_dense = keras.layers.EinsumDense(
                "bsd,kdh->bskh",
                output_shape=(None, self.heads, self.key_size),
                kernel_initializer=self.kernel_initializer,
                name="key",
            )
            self.key_dense.build(input_shape[1])

            self.value_dense = keras.layers.EinsumDense(
                "bsd,kdh->bskh",
                output_shape=(None, self.heads, self.head_size),
                kernel_initializer=self.kernel_initializer,
                name="value",
                
            )
            self.value_dense.build(input_shape[2])


            self.output_dense = keras.layers.EinsumDense(
                equation="btnh,nhd->btd",
                output_shape=(None, self.out_dim),
                kernel_initializer=self.kernel_initializer,
                name="attention_output",
            )
            self.output_dense.build(  
                (None, None, self.query_head, self.key_size)
            )
        else:
            self.q_dense = Dense(
                units=self.key_size * self.query_head,
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
                use_bias=self.o_bias,
                kernel_initializer=self.kernel_initializer
            )
        if self.attention_dropout:
            self.dropout=Dropout(self.attention_dropout)
        if self.rope_mode=='keras':
            self.rope = RotaryEmbedding(self.max_wavelength,self.scaling_factor)
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
        use_cache = kwargs.get('use_cache')
        if mask is not None:
            q_mask, v_mask = mask[0], mask[2]
        # 线性变换
        if self.use_EinsumDense:
            qw = self.query_dense(q)
            kw = self.key_dense(k)
            vw = self.value_dense(v)
        else:
            qw = self.q_dense(q)
            kw = self.k_dense(k)
            vw = self.v_dense(v)
            # 形状变换
            b,s = int_shape(qw)[:2]
            qw = ops.reshape(qw, [b,s,self.query_head, self.key_size])
            
            b,s = int_shape(vw)[:2]
            kw = ops.reshape(kw, [b,s,self.heads, self.key_size])
            vw = ops.reshape(vw, [b,s,self.heads, self.head_size])
        # Attention
        
        qkv_inputs = [qw, kw, vw] + inputs[3:]
        qv_masks = [q_mask, v_mask]
        
        o, a,cache = self.pay_attention_to(qkv_inputs, qv_masks, **kwargs)
        # 完成输出
        if self.use_EinsumDense:
            o = self.output_dense(o)
        elif self.flatten_o_dense:
            o = self.o_dense(ops.flatten(o, 2))
        else:
            b,s = ops.shape(o)[:-2]
            o = self.o_dense(ops.reshape(o, [b,s,-1]))
        # 返回结果
         

        if use_cache:
            return o,cache
        if self.return_attention_scores:
            return [o, a]
        else:
            return o

    def keras_apply_rope(self, x, start_index):
        """Rope rotate q or k."""
        x = self.rope(x, start_index=start_index)
        # Gemma uses a different layout for positional embeddings.
        # The transformation below ensures the embeddings are numerically
        # equivalent to the original gemma implementation.
        x = ops.reshape(
            ops.stack(ops.split(x, 2, axis=-1), axis=-1), ops.shape(x)
        )
        return x
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
        is_cache_update_index = kwargs.get('cache_update_index')
        use_cache = kwargs.get('use_cache')
        if a_bias:
            a_bias = inputs[n]
            n += 1
        
        if p_bias == 'rotary':
            if self.rope_mode=='su':
                qw, kw = apply_rotary_position_embeddings(inputs[n], qw, kw)
            elif self.rope_mode=='keras':
                qw = self.keras_apply_rope(qw, start_index=inputs[n+2] if use_cache else 0)
                kw = self.keras_apply_rope(kw, start_index=inputs[n+2] if use_cache else 0)
            else:
                raise('rope_mode must be su or keras')

            n += 1
        
        
        if use_cache:
            cache = inputs[n]
            n +=1
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if is_cache_update_index:
                cache_update_index = inputs[n]
                n += 1
                start = [0, cache_update_index, 0, 0]
            
                kw = ops.slice_update(key_cache, start, kw)
                vw = ops.slice_update(value_cache, start, vw)
                cache = ops.stack((kw, vw), axis=1)
            else:
                kw = key_cache
                vw = value_cache
        else:
            cache = None
        if enable_flashatt and not use_cache:
            is_causal = False
            if a_bias is not None :
                is_causal = True
            softmax_scale = 1.
            if self.attention_scale:
                softmax_scale = 1 / self.key_size**0.5
            o = flash_mha(qw,kw,vw,softmax_scale=softmax_scale, is_causal=is_causal)
            return o,[],cache
        # Attention
        q_shape = ops.shape(qw)
        b,s = q_shape[:-2]
        
        if self.GQA_mode=='gemma':
            qw = ops.reshape(
                qw,
                (
                    b,s,
                    self.heads,
                    self.query_head // self.heads,
                    q_shape[-1],
                ),
            )
            a = ops.einsum("btkgh,bskh->bkgts",qw,kw)
        else:
            if self.GQA_mode=='llama':
                num_key_value_groups = self.query_head // self.heads
                kw = ops.repeat(kw, repeats=num_key_value_groups, axis=2)
                vw = ops.repeat(vw, repeats=num_key_value_groups, axis=2)
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
            a = a * ops.cast(1/np.sqrt(self.key_size), dtype=qw.dtype)
        if a_bias is not None and ops.ndim(a_bias) == 3:
            a_bias = align(a_bias, [0, -2, -1], ops.ndim(a))
        
        
        A = attention_normalize(a, v_mask, -1, self.normalization, a_bias)
        
        if self.attention_dropout:
            A = self.dropout(A)

        # 完成输出
        if self.GQA_mode=='gemma':
            o = ops.einsum("bkgts,bskh->btkgh", A, vw)
            o = ops.reshape(o, (b, s, self.query_head, -1))
        else:
            o = ops.einsum('bhjk,bkhd->bjhd', A, vw)
        
        if p_bias == 'typical_relative':
            o = o + ops.einsum('bhjk,jkd->bjhd', A, position_bias)

        
        return o,a,cache

    
    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        for shape in input_shape:
            if len(shape)==5 and shape[1]==2:
                return [o_shape,shape]
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
            'max_wavelength':self.max_wavelength,
            'scaling_factor':self.scaling_factor ,
            'rope_mode':self.rope_mode,
            'o_bias':self.o_bias,
            'use_EinsumDense':self.use_EinsumDense,
            'query_head':self.query_head,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'GQA_mode':self.GQA_mode,
            'use_bias': self.use_bias,
            'normalization': self.normalization,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'flatten_o_dense':self.flatten_o_dense,
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
        if enable_flashatt and ops.shape(k)==ops.shape(v):
            z = self.pay_flash_attention_to(q,k,v, a_bias)
        else:
            z = self.pay_attention_to(q,k,v,mask, a_bias)   
        # 计算输出
        if self.self_attention==False and self.factorization:
            z = self.vW_dense(z)
        o = self.o_dense(u * z)
        return o
    def pay_flash_attention_to(self, q,k,v, a_bias):
        is_causal = False
        if a_bias is not None:
            is_causal = True
        softmax_scale = 1.
        if self.attention_scale:
            softmax_scale = 1 / self.key_size**0.5
        if ops.ndim(q)==3:
            k = ops.expand_dims(k,2)
            q = ops.expand_dims(q,2)
            v = ops.expand_dims(v,2)
        o = flash_mha(q,k,v,softmax_scale=softmax_scale, is_causal=is_causal)
        return ops.squeeze(o,2)
    def pay_attention_to(self, q,k,v,mask, a_bias):
        a = ops.einsum('bmd,bnd->bmn', q, k)
        if self.attention_scale:
            a = a / self.key_size**0.5
        A = attention_normalize(a, mask, -1, self.normalization, a_bias)
        if self.attention_dropout:
            A = self.dropout(A)
        try:
            z=ops.einsum('bmn,bnd->bmd', A, v)
        except:
            pass
        return z

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

class Gemma2Attention(keras.layers.Layer):
    """A cached grouped query attention layer."""

        
    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        kernel_initializer="glorot_uniform",
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        query_head_dim_normalize=True,
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.logit_soft_cap = logit_soft_cap
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.query_head_dim_normalize = query_head_dim_normalize
        self.dropout = dropout

        self._kernel_initializer = keras.initializers.get(
            kernel_initializer
        )
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.query_head_dim_normalize = query_head_dim_normalize

    def build(self, inputs_shape):
        self.hidden_dim = inputs_shape[-1]

        self.query_dense = keras.layers.EinsumDense(
            "btd,ndh->btnh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            "bsd,kdh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        self.value_dense = keras.layers.EinsumDense(
            "bsd,kdh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(inputs_shape)

        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        self.output_dense = keras.layers.EinsumDense(
            equation="btnh,nhd->btd",
            output_shape=(None, self.hidden_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )
        self.softmax = keras.layers.Softmax(dtype="float32")

        self.rope_layer = RotaryEmbedding(
            max_wavelength=10_000.0, dtype=self.dtype_policy
        )

        self.built = True

    def _apply_rope(self, x, start_index):
        """Rope rotate q or k."""
        x = self.rope_layer(x, start_index=start_index)
        # Gemma uses a different layout for positional embeddings.
        # The transformation below ensures the embeddings are numerically
        # equivalent to the original gemma implementation.
        x = ops.reshape(
            ops.stack(ops.split(x, 2, axis=-1), axis=-1), ops.shape(x)
        )
        return x

    def _compute_attention(
        self,
        q,
        k,
        v,
        attention_mask,
        training=False,
    ):
        if self.query_head_dim_normalize:
            query_normalization = 1 / np.sqrt(self.head_dim)
        else:
            query_normalization = 1 / np.sqrt(
                self.hidden_dim // self.num_query_heads
            )
        assert enable_flashatt==False #gemma2不支持flash attn
        q *= ops.cast(query_normalization, dtype=q.dtype)
        q_shape = ops.shape(q)
        q = ops.reshape(
            q,
            (
                *q_shape[:-2],
                self.num_key_value_heads,
                self.num_query_heads // self.num_key_value_heads,
                q_shape[-1],
            ),
        )
        b, q_len, _, _, h = ops.shape(q)

        attention_logits = ops.einsum("btkgh,bskh->bkgts", q, k)

        if self.logit_soft_cap is not None:
            attention_logits = ops.divide(attention_logits, self.logit_soft_cap)
            attention_logits = ops.multiply(
                ops.tanh(attention_logits), self.logit_soft_cap
            )

        if self.use_sliding_window_attention:
            all_ones = ops.ones_like(attention_mask)
            if keras.config.backend() == "tensorflow":
                import tensorflow as tf

                sliding_window_size = ops.minimum(
                    self.sliding_window_size - 1, q_len
                )
                sliding_window_size = ops.cast(
                    sliding_window_size, dtype="int32"
                )
                sliding_mask = tf.linalg.band_part(
                    all_ones, sliding_window_size - 1, sliding_window_size - 1
                )
                sliding_mask = ops.cast(sliding_mask, dtype="bool")
                bool_attention_mask = ops.cast(attention_mask, dtype="bool")
                attention_mask = tf.math.logical_and(
                    sliding_mask, bool_attention_mask
                )
            else:
                sliding_mask = ops.triu(
                    all_ones, -1 * self.sliding_window_size + 1
                ) * ops.tril(all_ones, self.sliding_window_size - 1)
                attention_mask = sliding_mask * attention_mask
        if ops.ndim(attention_mask)==2:
            attention_mask = ops.expand_dims(attention_mask,0)
        attention_mask = attention_mask[:, None, None, :, :]
        orig_dtype = attention_logits.dtype
        attention_softmax = self.softmax(attention_logits, mask=attention_mask)
        attention_softmax = ops.cast(attention_softmax, orig_dtype)

        if self.dropout:
            attention_softmax = self.dropout_layer(
                attention_softmax, training=training
            )

        results = ops.einsum("bkgts,bskh->btkgh", attention_softmax, v)
        return ops.reshape(results, (b, q_len, self.num_query_heads, h))

    def call(
        self,
        inputs,
        attention_mask=None,
        cache_update_index=0,
        training=False,
    ):
        if isinstance(inputs,list):
            x,cache = inputs[:]
        else:
            x = inputs
            cache = None
        query = self.query_dense(x)
        query = self._apply_rope(query, cache_update_index)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            key_update = self.key_dense(x)
            key_update = self._apply_rope(key_update, cache_update_index)
            value_update = self.value_dense(x)
            start = [0, cache_update_index, 0, 0]
            key = ops.slice_update(key_cache, start, key_update)
            value = ops.slice_update(value_cache, start, value_update)
            cache = ops.stack((key, value), axis=1)
        else:
            key = self.key_dense(x)
            key = self._apply_rope(key, cache_update_index)
            value = self.value_dense(x)

        attention_vec = self._compute_attention(
            query, key, value, attention_mask, training=training
        )

        # Wipe attn vec if there are no attended tokens.
        no_attended_tokens = ops.all(
            ops.equal(attention_mask, 0), axis=-1, keepdims=True
        )[..., None]
        attention_vec = ops.where(
            no_attended_tokens, ops.zeros_like(attention_vec), attention_vec
        )

        attention_output = self.output_dense(attention_vec)
        
        if cache is not None:
            return [attention_output, cache]
        return attention_output

    def compute_output_shape(self, input_shape):
        return input_shape
    
class DisentangledSelfAttention(keras.layers.Layer):
    """DisentangledSelfAttention layer.
    This code copy from keras_nlp lib
    This is an implementation of disentangled self-attention as described in the
    paper ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543).
    Effectively, this layer implements Multi-Head Self Attention with relative
    attention, i.e., to get the final attention score, we compute the
    content-to-position and position-to-content attention scores, and add these
    scores to the vanilla multi-head self-attention scores.

    Args:
        num_heads: int. Number of attention heads.
        hidden_dim: int. Hidden dimension of the input, i.e., `hidden_states`.
        max_position_embeddings: int. The maximum input
            sequence length. Defaults to `512`.
        bucket_size: int. The size of the relative position
            buckets. Generally equal to `max_sequence_length // 2`.
            Defaults to `256`.
        dropout: float. Dropout probability. Defaults to `0.1`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense layers.
            Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense layers.
            Defaults to `"zeros"`.
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        max_position_embeddings=512,
        bucket_size=256,
        dropout=0.1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Passed args.
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.max_position_embeddings = max_position_embeddings
        self.bucket_size = bucket_size
        self.dropout = dropout

        # Initializers.
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)

        # Derived args.
        self.attn_head_size = hidden_dim // num_heads

        # We have three types of attention - MHA, p2c and c2p.
        num_type_attn = 3
        self.scale_factor = 1.0 / math.sqrt(
            float(num_type_attn * self.attn_head_size)
        )

    def build(self, inputs_shape, rel_embeddings_shape=None):
        # Q, K, V linear layers.
        self._query_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)
        self._key_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)
        self._value_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        # Relative attention.
        self._position_dropout_layer = keras.layers.Dropout(
            self.dropout,
            dtype=self.dtype_policy,
        )

        self._attn_dropout_layer = keras.layers.Dropout(
            self.dropout,
            dtype=self.dtype_policy,
            name="attention_dropout",
        )
        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        # Output.
        self._output_dense = keras.layers.EinsumDense(
            equation="abc,cd->abd",
            output_shape=(None, self.hidden_dim),
            bias_axes="d",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build(inputs_shape)
        self.built = True

    def _get_common_kwargs_for_sublayer(self, use_bias=True):
        common_kwargs = {}

        kernel_initializer = self._kernel_initializer
        bias_initializer = self._bias_initializer

        common_kwargs["kernel_initializer"] = kernel_initializer
        if use_bias:
            common_kwargs["bias_initializer"] = bias_initializer

        return common_kwargs

    def _masked_softmax(self, attention_scores, attention_mask=None):
        """Normalizes the attention scores to probabilities using softmax.

        This implementation is the similar to the one present in
        `keras.layers.MultiHeadAttention`.
        """

        if attention_mask is not None:
            mask_expansion_axis = -3
            for _ in range(
                len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = ops.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax(attention_scores, attention_mask)

    def _compute_attention(
        self,
        query,
        key,
        value,
        rel_embeddings,
        attention_mask=None,
        training=None,
    ):
        """Computes the attention score and returns the attended outputs.

        This function computes vanilla MHA score, and relative attention scores
        (p2c and c2p). It then sums them up to get the final attention score,
        which is used to compute the attended outputs.
        """

        attention_scores = ops.einsum(
            "aecd,abcd->acbe",
            key,
            query,
        )
        attention_scores = ops.multiply(attention_scores, self.scale_factor)

        rel_embeddings = self._position_dropout_layer(
            rel_embeddings,
            training=training,
        )

        rel_attn_scores = self._compute_disentangled_attention(
            query=query,
            key=key,
            rel_embeddings=rel_embeddings,
        )

        if rel_attn_scores is not None:
            attention_scores += rel_attn_scores

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._attn_dropout_layer(
            attention_scores, training=training
        )
        attention_output = ops.einsum(
            "acbe,aecd->abcd", attention_scores, value
        )

        return attention_output, attention_scores

    def _make_log_bucket_position(self, rel_pos):
        dtype = rel_pos.dtype
        sign = ops.sign(rel_pos)
        mid = self.bucket_size // 2
        mid = ops.cast(mid, dtype=dtype)

        # If `rel_pos[i][j]` is out of bounds, assign value `mid`.
        abs_pos = ops.where(
            condition=(rel_pos < mid) & (rel_pos > -mid),
            x1=mid - 1,
            x2=ops.abs(rel_pos),
        )

        def _get_log_pos(abs_pos, mid):
            numerator = ops.log(abs_pos / mid)
            numerator = numerator * ops.cast(mid - 1, dtype=numerator.dtype)
            denominator = ops.log((self.max_position_embeddings - 1) / mid)
            val = ops.ceil(numerator / denominator)
            val = ops.cast(val, dtype=mid.dtype)
            val = val + mid
            return val

        log_pos = _get_log_pos(abs_pos, mid)

        bucket_pos = ops.where(
            condition=abs_pos <= mid,
            x1=rel_pos,
            x2=log_pos * sign,
        )
        bucket_pos = ops.cast(bucket_pos, dtype="int")

        return bucket_pos

    def _get_rel_pos(self, num_positions):
        ids = ops.arange(num_positions)
        ids = ops.cast(ids, dtype="int")
        query_ids = ops.expand_dims(ids, axis=-1)
        key_ids = ops.expand_dims(ids, axis=0)
        key_ids = ops.repeat(key_ids, repeats=num_positions, axis=0)

        rel_pos = query_ids - key_ids
        rel_pos = self._make_log_bucket_position(rel_pos)

        rel_pos = ops.expand_dims(ops.expand_dims(rel_pos, axis=0), axis=0)
        return rel_pos

    def _compute_disentangled_attention(
        self,
        query,
        key,
        rel_embeddings,
    ):
        """Computes relative attention scores (p2c and c2p)."""

        batch_size = ops.shape(query)[0]
        num_positions = ops.shape(query)[1]
        #如果你的最大长度小于自定义长度，应该考虑在输入的时候更改你的最大长度,比如输入参数max_position=2048
        assert num_positions is None or num_positions<=self.max_position_embeddings
        rel_pos = self._get_rel_pos(num_positions)

        rel_attn_span = self.bucket_size
        score = 0

        pos_query = self._query_dense(rel_embeddings)
        pos_key = self._key_dense(rel_embeddings)

        # c2p
        c2p_attn_scores = ops.einsum(
            "aecd,abcd->acbe",
            pos_key,
            query,
        )
        c2p_pos = ops.clip(rel_pos + rel_attn_span, 0, rel_attn_span * 2 - 1)
        c2p_pos = ops.broadcast_to(
            c2p_pos,
            shape=(
                batch_size,
                self.num_heads,
                num_positions,
                num_positions,
            ),
        )

        if keras.config.backend() == "tensorflow":
            # Work around dynamic shape bug on tensorflow backend.
            import tensorflow as tf

            c2p_attn_scores = tf.gather(
                c2p_attn_scores,
                indices=c2p_pos,
                batch_dims=3,
            )
        else:
            c2p_attn_scores = ops.take_along_axis(
                c2p_attn_scores,
                indices=c2p_pos,
                axis=3,
            )
        c2p_attn_scores = ops.multiply(c2p_attn_scores, self.scale_factor)
        score += c2p_attn_scores

        # p2c
        p2c_attn_scores = ops.einsum(
            "aecd,abcd->acbe",
            pos_query,
            key,
        )
        p2c_pos = ops.clip(-rel_pos + rel_attn_span, 0, rel_attn_span * 2 - 1)
        p2c_pos = ops.broadcast_to(
            p2c_pos,
            shape=(
                batch_size,
                self.num_heads,
                num_positions,
                num_positions,
            ),
        )
        if keras.config.backend() == "tensorflow":
            # Work around dynamic shape bug on tensorflow backend.
            import tensorflow as tf

            p2c_attn_scores = tf.gather(
                p2c_attn_scores,
                indices=p2c_pos,
                batch_dims=3,
            )
        else:
            p2c_attn_scores = ops.take_along_axis(
                p2c_attn_scores,
                indices=p2c_pos,
                axis=3,
            )
        p2c_attn_scores = ops.transpose(p2c_attn_scores, [0, 1, 3, 2])
        p2c_attn_scores = ops.multiply(p2c_attn_scores, self.scale_factor)
        score += p2c_attn_scores

        return score

    def call(
        self,
        inputs,
        rel_embeddings,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
    ):
        # `query`, `key`, `value` shape:
        # `(batch_size, sequence_length, num_heads, attn_head_size)`.
        query = self._query_dense(inputs)
        key = self._key_dense(inputs)
        value = self._value_dense(inputs)

        attention_output, attention_scores = self._compute_attention(
            query=query,
            key=key,
            value=value,
            rel_embeddings=rel_embeddings,
            attention_mask=attention_mask,
            training=training,
        )

        # Reshape `attention_output` to `(batch_size, sequence_length, hidden_dim)`.
        attention_output = ops.reshape(
            attention_output,
            [
                ops.shape(attention_output)[0],
                ops.shape(attention_output)[1],
                self.hidden_dim,
            ],
        )
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "bucket_size": self.bucket_size,
                "dropout": self.dropout,
                "kernel_initializer": keras.initializers.serialize(
                    self._kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self._bias_initializer
                ),
            }
        )
        return config