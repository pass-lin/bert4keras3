from bert4keras3.Models.Berts import *
class GPT(LM_Mask, BERT):
    """构建GPT模型
    链接：https://github.com/openai/finetune-transformer-lm
    """
    @insert_arguments(final_activation='softmax')
    @delete_arguments('with_pool', 'with_mlm')
    def __init__(self, **kwargs):
        super(GPT, self).__init__(**kwargs)

    def apply_embeddings(self, inputs):
        """GPT的embedding是token、position、segment三者embedding之和
        跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
        if self.custom_position_ids:
            p = inputs.pop(0)
        else:
            p = None

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
            inputs=self.simplify([x, p]),
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs

        # Language Model部分
        x = self.apply(
            inputs=x,
            layer=Embedding,
            arguments={'mode': 'dense'},
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Activation,
            activation=self.final_activation,
            name='LM-Activation'
        )

        return x

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(GPT, self).load_variable(checkpoint, name)
        if name == 'gpt/embeddings/word_embeddings':
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        """映射到TF版GPT权重格式
        """
        mapping = super(GPT, self).variable_mapping()
        mapping = {
            k: [
                i.replace('bert/', 'gpt/').replace('encoder', 'transformer')
                for i in v
            ]
            for k, v in mapping.items()
        }
        return mapping


class GPT2(GPT):
    """构建GPT2模型
    链接: https://github.com/openai/gpt-2
    """
    def get_inputs(self):
        """GPT2的输入是token_ids
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
        )
        return x_in

    def apply_embeddings(self, inputs):
        """GPT2的embedding是token、position两者embedding之和
        """
        x = inputs

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            name='Embedding-Position'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """GPT2的主体是基于Self-Attention的模块
        顺序：LN --> Att  --> Add --> LN --> FFN --> Add
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )
        x = self.apply(
            inputs=[x, x, x, attention_mask],
            layer=MultiHeadAttention,
            arguments={'a_bias': True},
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

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )
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
        return x

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Output-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Output-Dropout'
        )
        x = super(GPT2, self).apply_final_layers(x)

        return x

    def variable_mapping(self):
        """映射到TF版GPT2权重格式
        """
        mapping = super(GPT2, self).variable_mapping()
        mapping = {
            k: [i.replace('output/LayerNorm', 'input/LayerNorm') for i in v]
            for k, v in mapping.items()
        }
        mapping['Output-Norm'] = [
            'gpt/output/LayerNorm/beta',
            'gpt/output/LayerNorm/gamma',
        ]

        return mapping


class GPT2_ML(GPT):
    """构建GPT2_ML模型
    链接: https://github.com/imcaspar/gpt2-ml
    注意：GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的
         原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。
    """
    def get_inputs(self):
        """GPT2_ML的输入是token_ids
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
        )
        return x_in

    def apply_embeddings(self, inputs):
        """GPT2_ML的embedding是token、position两者embedding之和
        """
        x = inputs
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            name='Embedding-Position'
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """GPT2_ML的主体是基于Self-Attention的模块
        顺序：Att  --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x, attention_mask], {'a_bias': True}

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

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm-0' % feed_forward_name
        )
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
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm-1' % feed_forward_name
        )

        return x

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(GPT2_ML, self).load_variable(checkpoint, name)
        if name == 'newslm/embeddings/word_embed':
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        """映射到官方GPT2_ML权重格式
        """
        mapping = {
            'Embedding-Token': ['newslm/embeddings/word_embed'],
            'Embedding-Position': ['newslm/embeddings/pos_embed'],
            'Embedding-Norm': [
                'newslm/embeddings/LayerNorm_embed_norm/beta',
                'newslm/embeddings/LayerNorm_embed_norm/gamma',
            ],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'newslm/layer%02d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'query_layer/kernel',
                    prefix + 'query_layer/bias',
                    prefix + 'key_layer/kernel',
                    prefix + 'key_layer/bias',
                    prefix + 'value_layer/kernel',
                    prefix + 'value_layer/bias',
                    prefix + 'context_projection_layer/kernel',
                    prefix + 'context_projection_layer/bias',
                ],
                'Transformer-%d-FeedForward-Norm-0' % i: [
                    prefix + 'LayerNorm_mlp_ln0/beta',
                    prefix + 'LayerNorm_mlp_ln0/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/kernel',
                    prefix + 'intermediate/bias',
                    prefix + 'output/kernel',
                    prefix + 'output/bias',
                ],
                'Transformer-%d-FeedForward-Norm-1' % i: [
                    prefix + 'LayerNorm_mlp_ln1/beta',
                    prefix + 'LayerNorm_mlp_ln1/gamma',
                ],
            })

        return mapping

