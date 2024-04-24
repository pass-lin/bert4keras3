from bert4keras3.Models.Berts import *
class RoFormer(NEZHA):
    """旋转式位置编码的BERT模型
    链接：https://kexue.fm/archives/8265
    """
    def compute_cache_position_bias(self, inputs=None,self_cache_update_index=None,index=None):
        if self.cache_position_bias is None:

            self.cache_position_bias =self.apply(
                inputs=inputs,
                name='Embedding-Rotary-Position'
            )
        if inputs is not None:
            return None
        self.length_cache_position_bias = self.apply(
            inputs=self.cache_position_bias,
            layer=TakeLayer,
            axis=1,
            arguments={'index': self_cache_update_index},
            name='TakeLayer'
        )
        
        return self.length_cache_position_bias

    def apply_main_cache_layers(self, inputs, index,self_cache_update_index,
                                cross_cache_update_index=None,
                                attention_mask=None,position_bias=None,
            
                                ):
        x,caches = inputs[:]
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index

        # Self Attention
        xi, x  = x, [x, x, x,attention_mask, position_bias,caches[0],self_cache_update_index]
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
        x = self.apply(
            inputs=self.simplify([x, z]),
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            name='%s-Norm' % feed_forward_name
        )

        return [x,caches]
    def apply_main_layers(self, inputs, index):
        """RoFormer的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi, x = x, [x, x, x, position_bias]
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
            out_dim=self.hidden_size,
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
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
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
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def compute_position_bias(self, inputs=None):
        """Sinusoidal位置编码（直接返回）
        """
        if self.position_bias is None:

            if self.custom_position_ids:
                x = [inputs, self.inputs[2]]
            else:
                x = inputs

            self.position_bias = self.apply(
                inputs=x,
                layer=SinusoidalPositionEmbedding,
                output_dim=self.attention_key_size,
                merge_mode='zero',
                custom_position_ids=self.custom_position_ids,
                name='Embedding-Rotary-Position'
            )

        return self.position_bias


class RoFormerV2(RoFormer):
    """RoFormerV2
    改动：去掉bias，简化Norm，优化初始化等。
    """
    def initializer(self, shape, dtype=None, order=2, gain=1.0):
        """使用截断正态分布初始化
        """
        if shape[0] > 10000 or shape[0] < 10:
            hidden_size = shape[1]
        else:
            hidden_size = shape[0]
        gain *= self.num_hidden_layers**(-1. / order)
        stddev = 1.13684723 / hidden_size**0.5 * gain
        return keras.initializers.TruncatedNormal(stddev=stddev)(shape)

    def apply_embeddings(self, inputs):
        """RoFormerV2的embedding是token、segment两者embedding之和
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        if self.segment_vocab_size > 0:
            if self.shared_segment_embeddings:
                name = 'Embedding-Token'
            else:
                name = 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='Embedding-Norm'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                use_bias=False,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """RoFormerV2的主体是基于Self-Attention的模块
        顺序：Att  --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi = x
        x = [x, x, x, position_bias]
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
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            use_bias=False,
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
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=False,
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
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs

        if self.with_mlm:
            # 预测token概率部分
            if self.embedding_size != self.hidden_size:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.embedding_size,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='Output-Mapping'
                )
            x = self.apply(
                inputs=x,
                layer=Dropout,
                rate=self.dropout_rate,
                name='Output-MLM-Dropout'
            )
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={'mode': 'dense'},
                name='Embedding-Token'
            )
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation=mlm_activation,
                name='Output-MLM-Activation'
            )

        return x

    def variable_mapping(self):
        """删掉部分权重映射
        """
        mapping = super(RoFormerV2, self).variable_mapping()

        for k, v in mapping.items():
            v = [
                i for i in v
                if not string_matching(i, ['beta', 'gamma', 'bias'])
            ]
            mapping[k] = v

        return mapping

