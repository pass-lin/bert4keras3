from bert4keras3.Models.Roformers import *
class Gemma(LM_Mask,RoFormer):
    def __init__(self, with_lm=True,
                 max_wavelength=10_000.0,
                 scaling_factor=1.0,
                 use_dense_bias=False,
                 use_EinsumDense = True,
                 flatten_o_dense=False,
                 use_bias = False,
                 input_scale =True,
                 share_emebding=True,
                 rope_mode='keras',
                 **kwargs):
        super(Gemma, self).__init__(**kwargs)
        self.with_lm = with_lm
        self.max_wavelength = max_wavelength
        self.scaling_factor = scaling_factor
        self.rope_mode = rope_mode
        self.share_emebding = share_emebding
        self.use_dense_bias = use_dense_bias
        self.input_scale = input_scale
        self.flatten_o_dense = flatten_o_dense
        self.use_EinsumDense = use_EinsumDense
        self.use_bias = use_bias
        self.layer_norm_type = RMSNormalization
        self.ffn_type = GemmaFeedForward
    def apply_embeddings(self, inputs):
        inputs = inputs[:]
        
        x = inputs.pop(0)
        
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
        inputs = x
        
        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="untruncated_normal",
                seed=None,
            ),
            mask_zero=True,
            name='Embedding-Token'
        )
        
        if self.segment_vocab_size > 0:
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name='Embedding-Segment'
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )

        def mul(x):
            return x * ops.cast(ops.sqrt(self.hidden_size), x.dtype)
        if self.input_scale:
            x = self.apply(
                inputs=x,
                layer=Lambda,
                function=mul,
                name='Multiply'
            )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )

        
        return x
    

    
    def apply_main_layers(self, inputs, index):
        
        if isinstance(inputs,list):
            x = inputs[0]
        else:
            x = inputs

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)

        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi = x

        x = self.apply(
            inputs=x,
            layer=self.layer_norm_type,
            epsilon=1e-6,
            name='%s-Norm' % attention_name
        )
        x =  [x, x, x, position_bias]

        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(3, attention_mask)
        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            query_head=self.query_head,
            out_dim=self.hidden_size,
            use_EinsumDense = self.use_EinsumDense,
            use_bias= self.use_bias,
            normalization='softmax-fp32',
            flatten_o_dense = self.flatten_o_dense,
            rope_mode='keras',
            max_wavelength = self.max_wavelength,
            scaling_factor = self.scaling_factor,
            o_bias = self.o_bias,
            key_size=self.attention_key_size,
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
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
        # Feed Forward
        xi = x

        x = self.apply(
            inputs=x,
            layer=self.layer_norm_type,
            epsilon=1e-6,
            name='%s-Norm' % feed_forward_name
        )
        x = self.apply_ffn(x,feed_forward_name)
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        return x
    def apply_final_layers(self, inputs):
        """剩余部分
        """
        if isinstance(inputs,list):
            x = inputs[0]
        else:
            x = inputs

        x = self.apply(
            inputs=x,
            layer=self.layer_norm_type,
            epsilon=1e-6,
            name='Output-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Output-Dropout'
        )
        
        if self.with_lm:
            lm_activation = 'softmax' if self.with_lm is True else self.with_lm
            if self.share_emebding:
                x = self.apply(
                        inputs=x,
                        layer=Embedding,
                        arguments={'mode': 'dense'},
                        name='Embedding-Token'
                    )
                x = self.apply(
                        inputs=x,
                        layer=Activation,
                        activation=lm_activation,
                        name='Output-LM-Activation'
                    )
            else:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.vocab_size,
                    activation=lm_activation,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-LM'
                )

        
        return x
    def apply_ffn(self,x,feed_forward_name):
        x = self.apply(
            inputs=x,
            layer=self.ffn_type,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=self.use_dense_bias,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        return x
    def apply_main_cache_layers(self, inputs, index,self_cache_update_index,
                                cross_cache_update_index=None,
                                attention_mask=None,position_bias=None,
            
                                ):
        x,caches = inputs[:]
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index

        # Self Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            name='%s-Norm' % attention_name
        )
        x  = [x, x, x,attention_mask, position_bias,caches[0],self_cache_update_index]
        arguments = {'a_bias': True,'cache_update_index':True,'use_cache':True,'p_bias': 'rotary'}
        x,cache = self.apply(
            inputs=x,
            arguments=arguments,
            name=attention_name
        )
        
        caches[0] = cache
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            name='%s-Norm' % feed_forward_name
        )
        x = self.apply_ffn(x,feed_forward_name)
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        
        
        return [x,caches]

class Llama(Gemma):
    def __init__(self, input_scale =False,use_EinsumDense=False,
                 share_emebding=False,**kwargs):
        super(Llama, self).__init__(input_scale=input_scale,
                                    use_EinsumDense=use_EinsumDense,
                                    share_emebding=share_emebding,**kwargs)
        self.layer_norm_type = LlamaLayerNorm
        self.ffn_type = LLamaFeedForward