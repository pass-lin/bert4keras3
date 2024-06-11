# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:20:44 2024

@author: Administrator
"""

import numpy as np
from bert4keras3.backend import load_tf,keras,backlib,lora_model
from bert4keras3.layers import *
from bert4keras3.snippets import insert_arguments
from bert4keras3.snippets import delete_arguments
from bert4keras3.snippets import is_string, string_matching
from bert4keras3.snippets import orthogonally_resize
from keras.models import Model
import json

class Transformer(object):
    """模型基类
    """
    def __init__(
        self,
        vocab_size,  # 词表大小
        hidden_size,  # 编码维度
        num_hidden_layers,  # Transformer总层数
        num_attention_heads,  # Attention的头数
        intermediate_size,  # FeedForward的隐层维度
        hidden_act=None,  # FeedForward隐层的激活函数
        dropout_rate=None,  # Dropout比例
        attention_dropout_rate=None,  # Attention矩阵的Dropout比例
        embedding_size=None,  # 是否指定embedding_size
        attention_head_size=None,  # Attention中V的head_size
        attention_key_size=None,  # Attention中Q,K的head_size
        sequence_length=None,  # 是否固定序列长度
        keep_tokens=None,  # 要保留的词ID列表
        compound_tokens=None,  # 扩展Embedding
        residual_attention_scores=False,  # Attention矩阵加残差
        ignore_invalid_weights=False,  # 允许跳过不存在的权重
        autoresize_weights=False,  # 自动变换形状不匹配的权重
        layers=None,  # 外部传入的Keras层
        prefix=None,  # 层名前缀
        name=None,  # 模型名称
        segment_attention=False,
        o_bias=None,
        query_head=None,
        **kwargs
    ):
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.segment_attention = segment_attention
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.attention_dropout_rate = attention_dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights
        self.autoresize_weights = autoresize_weights
        self.layers = {} if layers is None else layers
        self.prefix = prefix or ''
        self.name = name
        self.built = False
        self.cache_position_bias = None
        self.cache_attention_bias= None
        self.single_head = False
        self.is_seq2seq = False
        self._seed_generators = []
        self.custom_position_ids = False
        self.o_bias = o_bias
        self.query_head = query_head
    def build(
        self,
        attention_caches=None,
        layer_norm_cond=None,
        layer_norm_cond_hidden_size=None,
        layer_norm_cond_hidden_act=None,
        additional_input_layers=None,
        **kwargs
    ):
        """模型构建函数
        attention_caches：为Attention的K,V的缓存序列字典，格式为
                         {Attention层名: [K缓存, V缓存]}；
        layer_norm_*系列参数：实现Conditional Layer Normalization时使用，
                            用来实现以“固定长度向量”为条件的条件Bert。
        """
        if self.built:
            return None
        # Input
        inputs = self.get_inputs()
        self.set_inputs(inputs, additional_input_layers)
        # Other
        self.attention_caches = attention_caches or {}
        self.layer_norm_conds = [
            layer_norm_cond,
            layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act or 'linear',
        ]
        
        outputs = self.call(inputs)
        self.set_outputs(outputs)
        # Model
        self.model = Model(self.inputs, self.outputs, name=self.name)
        self.built = True

    def call(self, inputs):
        """定义模型的执行流程
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs

    def prefixed(self, name):
        """给名字加前缀
        """
        if name is not None:
            return self.prefix + name

    def apply(self, inputs=None, layer=None, arguments=None, **kwargs):
        """通过apply调用层会自动重用同名层
        inputs: 上一层的输出；
        layer: 要调用的层类名；
        arguments: 传递给layer.call的参数；
        kwargs: 传递给层初始化的参数。
        """
        if layer is Dropout and self.dropout_rate == 0:
            return inputs

        if layer is MultiHeadAttention and self.residual_attention_scores:
            kwargs['return_attention_scores'] = True

        arguments = arguments or {}
        if layer is Lambda:
            kwargs['arguments'] = arguments
            arguments = {}

        name = self.prefixed(kwargs.get('name'))
        kwargs['name'] = name
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name
            self.layers[name] = layer

        if inputs is None:
            return self.layers[name]
        else:
            if isinstance(self.layers[name], MultiHeadAttention):
                if name in self.attention_caches:
                    # 如果检测到Cache的传入，那么自动在Key,Value处拼接起来
                    k_cache, v_cache = self.attention_caches[name]
                    k_name, v_name = name + '-Cached-Key', name + '-Cached-Value'
                    k = Concatenate1D(name=k_name)([k_cache, inputs[1]])
                    v = Concatenate1D(name=v_name)([v_cache, inputs[2]])
                    inputs = inputs[:1] + [k, v] + inputs[3:]
                if self.residual_attention_scores:
                    # 如果使用残差Attention矩阵，则给每个Attention矩阵加上前上一层的Attention
                    # 矩阵，这对应RealFormer设计（https://arxiv.org/abs/2012.11747）。目前
                    # 该实现还相对粗糙，可能欠缺通用性。
                    if self.attention_scores is not None:
                        if arguments.get('a_bias'):
                            a_bias = Add(name=name + '-Attention-Bias'
                                        )([inputs[3], self.attention_scores])
                            inputs = inputs[:3] + [a_bias] + inputs[4:]
                        else:
                            a_bias = self.attention_scores
                            inputs = inputs[:3] + [a_bias] + inputs[3:]
                        arguments['a_bias'] = True
                    o, a = self.layers[name](inputs, **arguments)
                    self.attention_scores = a
                    return o
            
            return self.layers[name](inputs, **arguments)

    def get_inputs(self):
        raise NotImplementedError

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs, index):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError

    def compute_attention_bias(self, inputs=None):
        """定义每一层的Attention Bias
        """
        return self.attention_bias

    def compute_position_bias(self, inputs=None):
        """定义每一层的Position Bias（一般相对位置编码用）
        """
        return self.position_bias

    def set_inputs(self, inputs, additional_input_layers=None):
        """设置input和inputs属性
        """
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        inputs = inputs[:]
        if additional_input_layers is not None:
            if not isinstance(additional_input_layers, list):
                additional_input_layers = [additional_input_layers]
            inputs.extend(additional_input_layers)

        self.inputs = inputs
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]

    def set_outputs(self, outputs):
        """设置output和outputs属性
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(stddev=0.02)

    def simplify(self, inputs):
        """将list中的None过滤掉
        """
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            inputs = inputs[0]

        return inputs

    def load_embeddings(self, embeddings):
        """处理Embedding层权重
        """
        embeddings = embeddings.astype(keras.mixed_precision.dtype_policy().name)  # 防止np.average报错

        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                if isinstance(item, list):
                    item = (item, [1] * len(item))
                ext_embeddings.append(
                    np.average(embeddings[item[0]], 0, item[1])
                )
            embeddings = np.concatenate([embeddings, ext_embeddings], 0)

        return embeddings

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        if isinstance(checkpoint, dict):
            return checkpoint[name]
        else:
            import tensorflow as tf
            return tf.train.load_variable(checkpoint, name)

    def create_variable(self, name, value, dtype=None):
        """创建一个变量
        """
        dtype = dtype or keras.mixed_precision.dtype_policy().name
        return K.variable(
            self.initializer(value.shape, dtype), dtype, name=name
        ), value

    def variable_mapping(self):
        """构建keras层与checkpoint的变量名之间的映射表
        """
        return {}
    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

       
        for layer, variables in mapping.items():
            weight_value_pairs = []
            layer = self.layers[layer]
            weights, values = [], []

            for w, v in zip(layer.trainable_weights, variables):  # 允许跳过不存在的权重
                try:
                    values.append(self.load_variable(checkpoint, v))
                    weights.append(w)
                except Exception as e:
                    if self.ignore_invalid_weights:
                        print('%s, but ignored.' % e.message)
                        weights.append(w)
                    else:
                        raise e

            for i, (w, v) in enumerate(zip(weights, values)):
                if v is not None:
                    w_shape, v_shape = int_shape(w), v.shape
                    if self.autoresize_weights and w_shape != v_shape:
                        v = orthogonally_resize(v, w_shape)
                        if isinstance(layer, MultiHeadAttention):
                            count = 2
                            if layer.use_bias:
                                count += 2
                            if layer.attention_scale and i < count:
                                scale = 1.0 * w_shape[-1] / v_shape[-1]
                                v = v * scale**0.25
                        if isinstance(layer, FeedForward):
                            count = 1
                            if layer.use_bias:
                                count += 1
                            if self.hidden_act in ['relu', 'leaky_relu']:
                                count -= 2
                            if i < count:
                                v *= np.sqrt(1.0 * w_shape[-1] / v_shape[-1])
                            else:
                                v *= np.sqrt(1.0 * v_shape[0] / w_shape[0])
                    weight_value_pairs.append(v)
                else:
                     weight_value_pairs.append(w.numpy())
            try:
                layer.set_weights(weight_value_pairs)
            except Exception as e:
                if self.ignore_invalid_weights:
                    print('%s, but ignored.' % e)
                else:
                    raise e
            
    def Search(self,inputs,k=1,mode='greedy'):
        if mode=='topp':
            return self.apply(
                inputs=inputs,
                layer=ToppSearch,
                k=k,
                dtype='float32',
                end_token=self.end_token,
                
            )
        elif mode=='topk':
            return self.apply(
                inputs=inputs,
                layer=TopkSearch,
                k=k,
                dtype='float32',
                end_token=self.end_token,
                
            )
        else:
            return self.apply(
                inputs=inputs,
                layer=GreedySearch,
                k=k,
                dtype='float32',
                end_token=self.end_token,
                
            )

    def compute_cache_position_bias(self, inputs=None,self_cache_update_index=None,index=None):
        return None

    def apply_main_cache_layers(self, inputs, index,self_cache_update_index,
                                cross_cache_update_index=None,
                                attention_mask=None,position_bias=None,
            
                                ):
        raise('this model not support cache model')
    def get_cache_inputs(self,lengths:list):
        raise('this model not support cache model')
    def get_custom_position_ids(self):
        return self.custom_position_ids
class LM_Mask(object):
    """定义下三角Attention Mask（语言模型用）
    """
        
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        if self.attention_bias is None:

            def lm_mask(s):
                seq_len = ops.shape(s)[1]
                idxs = ops.arange(0, seq_len)
                mask = idxs[None, :] <= idxs[:, None]
                return mask

            self.attention_bias = self.apply(
                inputs=self.inputs[0],
                layer=Lambda,
                function=lm_mask,
                dtype='float32',
                name='Attention-Mask'
            )

        return self.attention_bias
    def compute_cache_attention_bias(self, inputs=None,key=0,index=0):
        if self.cache_attention_bias==None:
            self.cache_attention_bias=self.apply(
                inputs=inputs[key],
                name='Attention-Mask'
            )
        else:
            return self.apply(
                inputs=self.cache_attention_bias,
                layer=TakeLayer,
                arguments={'index': index},
                axis=0,
                name='TakeLayer2'
            )
    def initial_cache(self,inputs):
        caches=[]
        class Initial_cache(Layer):
            def __init__(self, attention_key_size,num_attention_heads,single_head=False, **kwargs):
                super(Initial_cache, self).__init__(**kwargs)
                self.single_head =single_head
                self.attention_key_size=attention_key_size
                self.num_attention_heads=num_attention_heads
            def get_config(self):
                config = {
                    'single_head': self.single_head,
                    'num_attention_heads':self.num_attention_heads,
                    'attention_key_size':self.attention_key_size
                }
                base_config = super(Initial_cache, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
            def call(self,inputs, **kwargs):
                caches=[]
                for t in inputs:
                    if self.single_head:
                        cache_shape=[ops.shape(t)[0],2,ops.shape(t)[1],self.attention_key_size] 
                    else:
                        cache_shape=[ops.shape(t)[0],2,ops.shape(t)[1],self.num_attention_heads,self.attention_key_size]
                    caches.append(ops.zeros(cache_shape,dtype=self.compute_dtype))
                return caches
            def compute_output_shape(self, input_shape):
                shapes=[]
                for t in input_shape:
                    if self.single_head:
                        shapes.append([t[0],2,t[1],self.attention_key_size])      
                    else:
                        shapes.append([t[0],2,t[1],self.num_attention_heads,self.attention_key_size])   
                return shapes
        for _ in range(self.num_hidden_layers):
            caches.extend(self.apply(
                inputs=inputs,
                layer=Initial_cache,
                dtype='float32',
                single_head=self.single_head,
                num_attention_heads = self.num_attention_heads,
                attention_key_size = self.attention_key_size,
                name='Initial_caches'
            ))
        return caches
    def slice_inputs(self,inputs,key,index):
        return ops.expand_dims(ops.take(inputs[key],index,axis=1),1)
    def get_new_inputs(self,inputs,key,xs,index=None):
        return inputs[:key]+[xs]+inputs[key+1:]
    def cache_call(self,inputs:list,input_lengths:list,end_token,
                   search_mode='greedy',k=1,progress_print=True,index_bias=0):
        old_flag = self.custom_position_ids
        #print(keras.mixed_precision.dtype_policy())
        if self.is_seq2seq:
            caches = self.initial_cache([inputs[1],inputs[0]])
            key = 1
        else:
            caches = self.initial_cache(inputs[:1])
            key = 0
        x = inputs[key]

        class start_index(keras.Layer):
            def call(self,x):
                z = x!=0
                if index_bias>0:
                    t = ops.ones([ops.shape(z)[0],index_bias],dtype=z.dtype)
                    z = ops.slice_update(z,[0,0],t)
                return ops.cast(ops.max(ops.sum(z,-1))-1,'int32')
        
        
        length = input_lengths[key]
        self.cache_attention_bias=None
        self.cache_position_bias=None
        

        self.compute_cache_attention_bias(inputs,key,index=0)
        z = self.apply_embeddings(inputs)

        self.compute_cache_position_bias(z)

        self.end_token = end_token
        #initial inputs and cache

        

        if not isinstance(z,list):
            z = [z]
        j = len(caches)//self.num_hidden_layers
        
        for index in range(self.num_hidden_layers):
            layer_caches = caches[index*j:index*j+j]
            out=self.apply_main_cache_layers(z+[layer_caches], index,self_cache_update_index=ops.zeros([],'int32'),
                                        cross_cache_update_index=ops.zeros([],'int32'),
                                        attention_mask=self.cache_attention_bias,
                                        position_bias=self.cache_position_bias)
            z,cache = out[:-1],out[-1]
            
            caches[index*j:index*j+j]=cache
        
        
        
        index = self.apply(
            inputs=x,
            layer=start_index,
            dtype='float32',
            name='start_index'
            
        )
        
        def cond(inputs, caches, index , flags):
            cond1 = ops.less(index,length-1)
            cond2 = ops.logical_not(ops.all(ops.equal(inputs[key][:,index],end_token),-1))
            return ops.logical_and(cond1,cond2)
        
        def body(inputs, caches, index , flags):
            if progress_print:
                
                print('\r',index,end='')

            xs = self.slice_inputs(inputs,key,index)
            self.custom_position_ids = self.get_custom_position_ids()
            new_inputs = self.get_new_inputs(inputs,key,xs,index)
            
            if self.custom_position_ids:
                new_inputs += [ops.reshape(index,[-1,1])]

            z = self.apply_embeddings(new_inputs)
            if not isinstance(z,list):
                z = [z]
            attention_mask = self.compute_cache_attention_bias(index=index)

            position_bias = self.compute_cache_position_bias(self_cache_update_index = index) 

            for i in range(self.num_hidden_layers):
                
                layer_caches = caches[i*j:i*j+j]
                out=self.apply_main_cache_layers(z+[layer_caches], i,self_cache_update_index=index,
                                            cross_cache_update_index=None,
                                            attention_mask=attention_mask,
                                            position_bias=position_bias)
                
                z,cache = out[:-1],out[-1]
                
                caches[i*j:i*j+j]=cache
            

            o = self.apply_final_layers(z)
            index += 1

            search_in = [o,index,inputs[key],flags]

            inputs[key],flags = self.Search(search_in,k=k,mode=search_mode)

            return (inputs, caches, index , flags)
        num_hidden_layers = self.num_hidden_layers
        class WhileLayer(keras.Layer):
            def call(self, x):
                inputs, caches, index =  x[:]
                flags = ops.ones([ops.shape(caches[0])[0],1],dtype='bool')
                
                if backlib=='torch':
                    while cond(inputs, caches, index , flags):
                        inputs, caches, index , flags = body(inputs, caches, index , flags)
                    return (inputs, caches, index)
                
                outs=ops.while_loop(
                    cond,
                    body,
                    loop_vars=(inputs, caches, index , flags),
                    maximum_iterations=length-index                                                                                                                                                                                                                                                                                                                                                       ,
                )
                if progress_print:
                    print('\n')
                return outs[:3]
            def compute_output_shape(self, input_shape):
                return input_shape

        out=self.apply(
            inputs=(inputs, caches, index),
            layer=WhileLayer,
            name='WhileLayer',
            
        )

        self.custom_position_ids = old_flag
        return ops.cast(out[0][key],'int32')
    def build_cache_model(self,input_lengths:list,end_token,
                          search_mode='greedy',k=1,progress_print=False,index_bias=0):
        
        inputs=self.get_cache_inputs(input_lengths)

        out = self.cache_call(inputs=inputs,input_lengths=input_lengths,end_token=end_token,
                       search_mode=search_mode,k=k,progress_print=progress_print,index_bias=index_bias)

        model = keras.Model(inputs,out)
        inputs = []
        for modelin in model.inputs: 
            shape=keras.ops.shape(modelin)
            shape=[1 if t==None else t for t in shape]
            inputs.append(ops.convert_to_tensor(np.ones(shape),modelin.dtype))
        self.cache_call(inputs=inputs,input_lengths=input_lengths,end_token=end_token,
                       search_mode=search_mode,k=k,progress_print=progress_print,index_bias=index_bias)
        
        return model
class UniLM_Mask(LM_Mask):
    """定义UniLM的Attention Mask（Seq2Seq模型用）
    其中source和target的分区，由segment_ids来表示。
    UniLM: https://arxiv.org/abs/1905.03197
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        if self.attention_bias is None:

            if self.segment_attention:
                def unilm_mask(s):
                    mask = s[:, None, :] <= s[:, :, None]
                    return mask
            else:
                def unilm_mask(s):
                    idxs = ops.cumsum(s, axis=1)
                    mask = idxs[:, None, :] <= idxs[:, :, None]
                    return mask
            
            self.attention_bias = self.apply(
                inputs=self.inputs[1],
                layer=Lambda,
                function=unilm_mask,
                dtype='float32',
                name='Attention-Mask'
            )

        return self.attention_bias
    def slice_inputs(self,inputs,key,index):
        def take(inputs,key,index):
            return ops.expand_dims(ops.take(inputs[key],index,axis=1),1)
        return [take(inputs,key,index),take(inputs,key+1,index)]
    def get_new_inputs(self,inputs,key,xs,index=None):
        if self.custom_position_ids:
            return inputs[:key]+xs+[ops.expand_dims(index,0)]+inputs[key+2:]
        return inputs[:key]+xs+inputs[key+2:]
    def compute_cache_attention_bias(self, inputs=None,key=0,index=0):
        
        if self.cache_attention_bias==None:
            self.cache_attention_bias=self.apply(
                inputs=inputs[key+1],
                name='Attention-Mask'
            )
        else:
            return self.apply(
                inputs=self.cache_attention_bias,
                layer=TakeLayer,
                axis=1,
                arguments={'index': index},
                name='TakeLayer'
            )


        return self.attention_bias
    def slice_inputs(self,inputs,key,index):
        def take(inputs,key,index):
            return ops.expand_dims(ops.take(inputs[key],index,axis=1),1)
        return [take(inputs,key,index),take(inputs,key+1,index)]
    def get_new_inputs(self,inputs,key,xs,index=None):
        if self.custom_position_ids:
            return inputs[:key]+xs+[ops.expand_dims(index,0)]+inputs[key+2:]
        return inputs[:key]+xs+inputs[key+2:]
    def compute_cache_attention_bias(self, inputs=None,key=0,index=0):
        
        if self.cache_attention_bias==None:
            self.cache_attention_bias=self.apply(
                inputs=inputs[key+1],
                name='Attention-Mask'
            )
        else:
            return self.apply(
                inputs=self.cache_attention_bias,
                layer=TakeLayer,
                axis=1,
                arguments={'index': index},
                name='TakeLayer'
            )