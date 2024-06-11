from bert4keras3.Models.Roformers import *
class T5_Base(Transformer):
    """Google的T5模型（基类）
    注意T5有两个版本，一开始放出来的版本称为t5.1.0，而后来放出了一个升级
    版本称为t5.1.1，两者结构略有不同，包括后来放出来的多国语言版T5也采用
    了t5.1.1的结构。
    t5.1.0: https://github.com/google-research/text-to-text-transfer-transformer
    t5.1.1: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md#t511
    multilingual-t5: https://github.com/google-research/multilingual-t5
    """
    @insert_arguments(version='t5.1.0')
    def __init__(self, **kwargs):
        super(T5_Base, self).__init__(**kwargs)
        self.p_bias = 't5_relative'
    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(T5_Base, self).load_variable(checkpoint, name)
        if name == 'shared/embedding':
            return self.load_embeddings(variable)
        elif name == 'decoder/logits/kernel':
            return self.load_embeddings(variable.T).T
        elif 'relative_attention_bias' in name:
            return variable.T
        else:
            return variable

    def create_variable(self, name, value, dtype=None):
        """在tensorflow中创建一个变量
        """
        if 'relative_attention_bias' in name:
            value = value.T
        return super(T5_Base, self).create_variable(name, value, dtype)

    def variable_mapping(self):
        """映射到官方T5权重格式
        """
        mapping = {
            'Embedding-Token': ['shared/embedding'],
            'Encoder-Embedding-Relative-Position': [
                'encoder/block_000/layer_000/SelfAttention/relative_attention_bias'
            ],
            'Encoder-Output-Norm': ['encoder/final_layer_norm/scale'],
            'Decoder-Embedding-Relative-Position': [
                'decoder/block_000/layer_000/SelfAttention/relative_attention_bias',
            ],
            'Decoder-Output-Norm': ['decoder/final_layer_norm/scale'],
        }

        for i in range(self.num_hidden_layers):
            # Encoder主体
            prefix = 'encoder/block_%03d/' % i
            mapping.update({
                'Encoder-Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'layer_000/SelfAttention/q',
                    prefix + 'layer_000/SelfAttention/k',
                    prefix + 'layer_000/SelfAttention/v',
                    prefix + 'layer_000/SelfAttention/o',
                ],
                'Encoder-Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'layer_000/layer_norm/scale',
                ],
                'Encoder-Transformer-%d-FeedForward' % i: [
                    prefix + 'layer_001/DenseReluDense/wi/kernel',
                    prefix + 'layer_001/DenseReluDense/wo/kernel',
                ],
                'Encoder-Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'layer_001/layer_norm/scale',
                ],
            })
            # Decoder主体
            prefix = 'decoder/block_%03d/' % i
            mapping.update({
                'Decoder-Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'layer_000/SelfAttention/q',
                    prefix + 'layer_000/SelfAttention/k',
                    prefix + 'layer_000/SelfAttention/v',
                    prefix + 'layer_000/SelfAttention/o',
                ],
                'Decoder-Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'layer_000/layer_norm/scale',
                ],
                'Decoder-Transformer-%d-MultiHeadCrossAttention' % i: [
                    prefix + 'layer_001/EncDecAttention/q',
                    prefix + 'layer_001/EncDecAttention/k',
                    prefix + 'layer_001/EncDecAttention/v',
                    prefix + 'layer_001/EncDecAttention/o',
                ],
                'Decoder-Transformer-%d-MultiHeadCrossAttention-Norm' % i: [
                    prefix + 'layer_001/layer_norm/scale',
                ],
                'Decoder-Transformer-%d-FeedForward' % i: [
                    prefix + 'layer_002/DenseReluDense/wi/kernel',
                    prefix + 'layer_002/DenseReluDense/wo/kernel',
                ],
                'Decoder-Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'layer_002/layer_norm/scale',
                ],
            })

        if self.version.endswith('t5.1.1'):
            mapping['Decoder-Output-LM'] = ['decoder/logits/kernel']
            for i in range(self.num_hidden_layers):
                for layer in [
                    'Encoder-Transformer-%d-FeedForward' % i,
                    'Decoder-Transformer-%d-FeedForward' % i
                ]:
                    mapping[layer] = [
                        mapping[layer][0][:-7] + '_0' + mapping[layer][0][-7:],
                        mapping[layer][0][:-7] + '_1' + mapping[layer][0][-7:],
                        mapping[layer][1]
                    ]
            if self.version == 'mt5.1.1':
                mapping['Encoder-Output-Norm'] = ['encoder/rms_norm/scale']
                mapping['Decoder-Output-Norm'] = ['decoder/rms_norm/scale']
                mapping = {
                    k: [i.replace('layer_norm', 'rms_norm') for i in v]
                    for k, v in mapping.items()
                }

        return mapping


class T5_Encoder(T5_Base):
    """Google的T5模型（Encoder）
    """
    def __init__(self,segment_size=0, **kwargs):
        super(T5_Encoder, self).__init__(**kwargs)
        self.segment_vocab_size=segment_size
        

    def get_inputs(self):
        """T5的Encoder的输入只有token_ids
        """
        x_in = self.apply(
            layer=Input,
            shape=(self.sequence_length,),
            name='Encoder-Input-Token',dtype='int32'
        )
        if self.segment_vocab_size > 0:
            s_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Segment-Input-Token',dtype='int32'
            )
            return [x_in,s_in]
        return x_in

    def apply_embeddings(self, inputs):
        """T5的embedding只有token embedding，
        并把relative position embedding准备好，待attention使用。
        """
        if type(inputs)==list:
            x,s = inputs[:]
        else:
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
        if self.segment_vocab_size > 0:
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer='zeros',
                name='Embedding-Segment'
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Encoder-Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Encoder-Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """T5的Encoder的主体是基于Self-Attention的模块
        顺序：LN --> Att --> Add --> LN --> FFN --> Add
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Encoder-Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Encoder-Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            zero_mean=False,
            offset=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )
        x = self.apply(
            inputs=[x, x, x, position_bias],
            layer=MultiHeadAttention,
            arguments={'p_bias': self.p_bias},
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
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
            zero_mean=False,
            offset=False,
            epsilon=1e-6,
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

        return x

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            zero_mean=False,
            offset=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Encoder-Output-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Encoder-Output-Dropout'
        )

        return x

    def compute_position_bias(self, inputs=None):
        """T5相对位置编码
        """
        if self.position_bias is None:

            x = inputs
            p = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=True,
                embeddings_initializer=self.initializer,
                name='Encoder-Embedding-Relative-Position'
            )
            self.position_bias = p

        return self.position_bias
    

class T5_Decoder(LM_Mask, T5_Base):
    """Google的T5模型（Decoder）
    """
    def __init__(self, with_lm=True, cross_position_bias=True,logit_scale=True, decoder_sequence_length=None,**kwargs):
        super(T5_Decoder, self).__init__(**kwargs)
        self.with_lm = with_lm
        self.cross_position_bias = cross_position_bias
        self.logit_scale=logit_scale
        self.is_seq2seq = True
        self.decoder_sequence_length=decoder_sequence_length
    def get_inputs(self):
        """T5的Decoder的输入为context序列和token_ids
        """
        c_in = self.apply(
            layer=Input,
            shape=(self.sequence_length, self.hidden_size),
            name='Input-Context'
        )
        x_in = self.apply(
            layer=Input,
            shape=(self.decoder_sequence_length,),
            name='Decoder-Input-Token',dtype='int32'
        )
        return [c_in, x_in]

    def apply_embeddings(self, inputs):
        """T5的embedding只有token embedding，
        并把relative position embedding准备好，待attention使用。
        """
        c, x = inputs

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
            layer=Dropout,
            rate=self.dropout_rate,
            name='Decoder-Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Decoder-Embedding-Mapping'
            )

        return [c, x]

    def apply_main_layers(self, inputs, index):
        """T5的Decoder主体是基于Self-Attention、Cross-Attention的模块
        顺序：LN --> Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add
        """
        c, x = inputs
        z = self.layer_norm_conds[0]

        self_attention_name = 'Decoder-Transformer-%d-MultiHeadSelfAttention' % index
        cross_attention_name = 'Decoder-Transformer-%d-MultiHeadCrossAttention' % index
        feed_forward_name = 'Decoder-Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias([x, c])

        # Self Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            zero_mean=False,
            offset=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % self_attention_name
        )
        p = position_bias
        if self.p_bias=='t5_relative':
            p = position_bias[0]
        x = self.apply(
            inputs=[x, x, x, attention_mask,p ],
            layer=MultiHeadAttention,
            arguments={
                'a_bias': True,
                'p_bias': self.p_bias
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=self_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % self_attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % self_attention_name
        )

        # Cross Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            zero_mean=False,
            offset=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % cross_attention_name
        )
        if self.cross_position_bias:
            inputs = [x, c, c, position_bias[1]]
            arguments = {'a_bias': None, 'p_bias': self.p_bias}
        else:
            inputs = [x, c, c]
            arguments = {'a_bias': None, 'p_bias': None}
        x = self.apply(
            inputs=inputs,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=cross_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % cross_attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % cross_attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            zero_mean=False,
            offset=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )
        x = self.apply_ffn_layer(x,feed_forward_name)
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )

        return [c, x]
    def apply_ffn_layer(self,x,feed_forward_name):
        return self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=False,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
    def apply_final_layers(self, inputs):
        """剩余部分
        """
        c, x = inputs
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            zero_mean=False,
            offset=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Decoder-Output-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Decoder-Output-Dropout'
        )
        if self.logit_scale:
            x = self.apply(
                inputs=x,
                layer=ScaleOffset,
                scale=self.hidden_size**(-0.5),
                offset=False,
                name='Decoder-Output-Scale'
            )

        if self.with_lm:
            # 预测token概率部分
            if self.embedding_size != self.hidden_size:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.embedding_size,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-Mapping'
                )
            lm_activation = 'softmax' if self.with_lm is True else self.with_lm
            if self.version == 't5.1.0':
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
                    name='Decoder-Output-LM-Activation'
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

    def compute_attention_bias(self, inputs=None):
        """修改LM Mask的序列长度（从 self.inputs[0] 改为 self.inputs[1] ）
        """
        old_inputs = self.inputs[:]
        self.inputs = [old_inputs[1]]
        mask = super(T5_Decoder, self).compute_attention_bias(inputs)
        self.inputs = old_inputs
        return mask

    def compute_position_bias(self, inputs=None):
        """T5相对位置编码
        """
        if self.position_bias is None:

            x, c = inputs
            p1 = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position'
            )
            p2 = self.apply(
                inputs=[x, c],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position'
            )
            self.position_bias = (p1, p2)

        return self.position_bias
    def get_cache_inputs(self,lengths:list):
        """Misaka的Decoder的输入为context序列和token_ids
        """
        c_in = self.apply(
            layer=Input,
            shape=(lengths[0], self.hidden_size),
            name='Input-Context-cache-'+str(lengths[1])
        )
        x_in = self.apply(
            layer=Input,
            shape=[lengths[1]],dtype='int32',
            name='Decoder-Input-Token-cache-'+str(lengths[1])
        )
        return [c_in, x_in] 
    def compute_attention_bias(self, inputs=None):
        """修改LM Mask的序列长度（从 self.inputs[0] 改为 self.inputs[1] ）
        """
        old_inputs = self.inputs[:]
        self.inputs = [old_inputs[1]]
        mask = super(T5_Decoder, self).compute_attention_bias(inputs)
        self.inputs = old_inputs
        return mask

    def compute_position_bias(self, inputs=None):
        """T5相对位置编码
        """
        if self.position_bias is None:

            x, c = inputs
            p1 = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position'
            )
            p2 = self.apply(
                inputs=[x, c],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position'
            )
            self.position_bias = (p1, p2)

        return self.position_bias
    def compute_cache_position_bias(self, inputs=None,self_cache_update_index=None,index=None):
        """T5相对位置编码
        """
        if self.cache_position_bias is None:

            c,x = inputs
            p1 = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position'
            )
            p2 = self.apply(
                inputs=[x, c],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position'
            )
            self.cache_position_bias = (p1, p2)
        if inputs!=None:
            return None
        p1,p2=self.cache_position_bias


        p1 = self.apply(
            inputs=p1,
            layer=TakeLayer,
            arguments={'index': self_cache_update_index},
            name='TakeLayer'
        )
        
        p2 = self.apply(
            inputs=p2,
            layer=TakeLayer,
            arguments={'index': self_cache_update_index},
            name='TakeLayer'
        )
        self.length_cache_position_bias = [p1,p2]
        
        return self.length_cache_position_bias

    def apply_main_cache_layers(self, inputs, index,self_cache_update_index,
                                cross_cache_update_index=None,
                                attention_mask=None,position_bias=None,
            
                                ):
        """T5的Decoder主体是基于Self-Attention、Cross-Attention的模块
        顺序：LN --> Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add
        """
        c, x ,caches = inputs
        z = self.layer_norm_conds[0]

        self_attention_name = 'Decoder-Transformer-%d-MultiHeadSelfAttention' % index
        cross_attention_name = 'Decoder-Transformer-%d-MultiHeadCrossAttention' % index
        feed_forward_name = 'Decoder-Transformer-%d-FeedForward' % index

        
        # Self Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            name='%s-Norm' % self_attention_name
        )
        arguments={
                'a_bias': True,
                'p_bias': self.p_bias,
                'cache_update_index':True,
                'use_cache':True,
            }
        p = position_bias
        if self.p_bias == 't5_relative':
            p = position_bias[0]
            inputs = [x, x, x, attention_mask,caches[0],self_cache_update_index,p]
        else:
            inputs = [x, x, x, attention_mask,p,caches[0],self_cache_update_index]
        x,cache_self = self.apply(
            inputs=inputs,
            arguments=arguments,
            name=self_attention_name
        )
        caches[0]=cache_self
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % self_attention_name
        )

        # Cross Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            name='%s-Norm' % cross_attention_name
        )
        if self.cross_position_bias:
            inputs = [x, c, c, position_bias[1]]
            arguments = {'a_bias': None, 'p_bias': self.p_bias}
        else:
            inputs = [x, c, c]
            arguments = {'a_bias': None, 'p_bias': None}
        arguments['use_cache']=True
        
        inputs.insert(3,caches[1])
        if cross_cache_update_index is None:
            arguments['cache_update_index']=False
        else:
            arguments['cache_update_index']=True
            inputs.insert(4,cross_cache_update_index)

        x ,cross_cache = self.apply(
                inputs=inputs,
                arguments=arguments,
                name=cross_attention_name
            )
        caches[1]=cross_cache
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % cross_attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            name='%s-Norm' % feed_forward_name
        )
        x = self.apply_ffn_layer(x,feed_forward_name)
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )

        return [c, x ,caches]
class T5(T5_Base):
    """Google的T5模型（Encoder-Decoder）
    """
    def __init__(self, **kwargs):
        super(T5, self).__init__(**kwargs)
        kwargs['layers'] = self.layers
        e_name, d_name = 'Encoder', 'Decoder'
        if 'name' in kwargs:
            e_name = '%s_%s' % (kwargs['name'], e_name)
            d_name = '%s_%s' % (kwargs['name'], d_name)
            del kwargs['name']  # 防止重复传参
        self._encoder = T5_Encoder(name=e_name, **kwargs)
        self._decoder = T5_Decoder(name=d_name, **kwargs)

    def build(self, **kwargs):
        """同时构建Encoder和Decoder
        """
        self._encoder.build(**kwargs)
        self._decoder.build(**kwargs)
        self._decoder.position_bias = None  # 下面call时将重新初始化
        self.encoder = self._encoder.model
        self.decoder = self._decoder.model
        self.inputs = self.encoder.inputs + self.decoder.inputs[1:]
        self.outputs = self._decoder.call(
            self.encoder.outputs + self.decoder.inputs[1:]
        )
        self.model = Model(self.inputs, self.outputs)
    def build_cache_model(self,input_lengths:list,end_token,search_mode='greedy',k=1,progress_print=False,index_bias=0):
        self.cache_decoder = self._decoder.build_cache_model(input_lengths,end_token,
                                                    search_mode,k,
                                                    progress_print,
                                                    index_bias)
        y = self.cache_decoder([self.encoder.output,self.cache_decoder.inputs[1]])
        self.cache_t5 = keras.Model(self.encoder.inputs[:]+self.cache_decoder.inputs[1:],y)

        return self.cache_t5
class MisakaT5(T5):
    """Google的T5模型（Encoder-Decoder）
    """
    def __init__(self, **kwargs):
        super(T5, self).__init__(**kwargs)
        kwargs['layers'] = self.layers
        e_name, d_name = 'Encoder', 'Decoder'
        if 'name' in kwargs:
            e_name = '%s_%s' % (kwargs['name'], e_name)
            d_name = '%s_%s' % (kwargs['name'], d_name)
            del kwargs['name']  # 防止重复传参
        self._encoder = MisakaT5_Encoder(name=e_name, **kwargs)
        self._decoder = MisakaT5_Decoder(name=d_name, **kwargs)

class MisakaT5_Encoder(T5_Encoder):
    def __init__(self, **kwargs):
        super( MisakaT5_Encoder, self).__init__(**kwargs)
        self.p_bias = 'rotary'
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
class MisakaT5_Decoder(T5_Decoder):
    def __init__(self, **kwargs):
        super( MisakaT5_Decoder, self).__init__(**kwargs)
        self.p_bias = 'rotary'
    def compute_cache_position_bias(self, inputs=None,self_cache_update_index=None,index=None):
        if self.cache_position_bias is None:

            self.cache_position_bias =self.apply(
                inputs=inputs[1],
                name='Embedding-Rotary-Position'
            )
        if inputs!=None:
            return None
        self.length_cache_position_bias = self.apply(
            inputs=self.cache_position_bias,
            layer=TakeLayer,
            axis=1,
            arguments={'index': self_cache_update_index},
            name='TakeLayer'
        )
        
        return self.length_cache_position_bias
    def compute_position_bias(self, inputs=None):
        """Sinusoidal位置编码（直接返回）
        """
        
        if self.position_bias is None:

            x, c = inputs[:]
           
            self.position_bias = self.apply(
                inputs=x,
                layer=SinusoidalPositionEmbedding,
                output_dim=self.attention_key_size,
                merge_mode='zero',
                custom_position_ids=self.custom_position_ids,
                name='Embedding-Rotary-Position'
            )

        return self.position_bias
        
