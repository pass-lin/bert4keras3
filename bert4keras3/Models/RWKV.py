from bert4keras3.Layers_add.Embeddings import Embedding
from bert4keras3.backend import ops,keras,lora_model
from bert4keras3.Layers_add.Rwkv_layer import *
from bert4keras3.transformers import *
class RWKV6(Transformer):
    def __init__(self,decomposer_size,with_lm=True,
                 time_decay_size = 64,
                 input_state=False,#输入每一层的state
                 output_state=False,#输出每一层的state
                 **kwargs):
        super().__init__(**kwargs)
        from rwkv6_keras_operator import RWKVKernelOperator
        self.decomposer_size = decomposer_size
        self.with_lm = with_lm
        self.input_state = input_state
        self.rwkv_kernal = RWKVKernelOperator(self.attention_head_size, self.sequence_length)
        self.output_state = output_state
        self.states = []
        self.time_decay_size = time_decay_size
        self.wkv_dtype = 'float32'
            
    def get_inputs(self):
        x_in = self.apply(
            layer=keras.Input, shape=(self.sequence_length,), name='Input-Token',dtype='int32'
        )
        return [x_in]
    def take_last_x(self,x,last_x_index=None):
        class Take(keras.Layer):
            def call(self,x,index=None):
                
                if index is not None:
                    o = ops.take(x,index-1,1)[:,None]
                    return o
                #这里如果x的shape不是[B,1,C]就会出问题
                assert int_shape(x)[1]==1
                return x

            def compute_output_shape(self, input_shape):
                input_shape = list(input_shape)
                input_shape[1] = 1
                return input_shape
        return self.apply(
                inputs=x,
                layer=Take,
                arguments={'index':last_x_index},
                name='TakeLayer-X'
            )
    def set_inputs(self, inputs, additional_input_layers=None):
        
        super().set_inputs(inputs, additional_input_layers)
        if self.input_state:
            states =self.apply(
                    layer=keras.Input, shape=(self.num_hidden_layers,
                        self.hidden_size // self.attention_head_size,
                        self.attention_head_size,self.attention_head_size), 
                        name='Stack-Statess',dtype=self.wkv_dtype
                )
            self.inputs.append(states)
    def get_mask(self,x):
        if self.attention_bias is None:

            def mask(s):
                return ops.not_equal(s,0)

            self.attention_bias = self.apply(
                inputs=x,
                layer=keras.layers.Lambda,
                function=mask,
                name='RKKV-Mask'
            )

        return self.attention_bias
    def apply_embeddings(self, inputs):
        x = inputs.pop(0)
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
            inputs = x,
            layer = keras.layers.LayerNormalization,
            epsilon=1e-5,
            name='Embedding-LN'
        )
        return x
    def apply_main_layers(self, inputs, index):
        """RWKV的主体是基于Self-Attention的模块
        顺序：LN-->Time-Mix --> Add  --> LN --> Channal-Mix --> Add
        """
        x = inputs
        z = self.layer_norm_conds[0]

        timemix_name = 'RWKV-%d-TimeMix' % index
        channelmix_name = 'RWKV-%d-ChannelMix' % index

        xi = x

        x = self.apply(
            inputs = x,
            layer = keras.layers.LayerNormalization,
            epsilon=1e-5,
            name='%s-Norm' % timemix_name
        )
        mask = self.get_mask(self.inputs[0])
        arguments = {'with_state':False,'initial_state':False}
        inputs = [x,mask]
        if self.output_state:
            arguments['with_state']=True
        if self.input_state:
            arguments['initial_state']=True
            inputs.append(self.inputs[-1][:,index])
        
        out = self.apply(
            inputs = inputs,
            layer = TimeMix,
            layer_idx = index,
            arguments=arguments,
            rwkv_kernel = self.rwkv_kernal,
            hidden_size = self.hidden_size,
            time_decay_size = self.time_decay_size,
            head_size = self.attention_head_size,
            decomposer_size = self.decomposer_size,
            name = timemix_name
        )
        x = out[0]
        if self.output_state:
            self.states.append(out[1])

        x = self.apply(
            inputs=x,
            layer=keras.layers.Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % timemix_name
        )

        x = self.apply(
            inputs=[xi, x], layer=keras.layers.Add, name='%s-Add' % timemix_name
        )

        xi = x

        x = self.apply(
            inputs = x,
            layer = keras.layers.LayerNormalization,
            epsilon=1e-5,
            name='%s-Norm' % channelmix_name,dtype='float32'
        )
        
        x = self.apply_ffn_layer(x,channelmix_name)
        x = self.apply(
            inputs=x,
            layer=keras.layers.Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % channelmix_name
        )

        x = self.apply(
            inputs=[xi, x], layer=keras.layers.Add, name='%s-Add' % channelmix_name
        )
        return x
    def apply_ffn_layer(self,x,name):
        return self.apply(
            inputs = x,
            layer = ChannelMix,
            hidden_size = self.hidden_size,
            expand_size = self.intermediate_size,
            name = name,
        )
    def apply_main_cache_layers(self, x, index,initial_state=None,mask=None,rnn_mode=False,last_x=None,last_x_index=None):
        """RWKV的主体是基于Self-Attention的模块
        顺序：LN-->Time-Mix --> Add  --> LN --> Channal-Mix --> Add
        """

        timemix_name = 'RWKV-%d-TimeMix' % index
        channelmix_name = 'RWKV-%d-ChannelMix' % index

        xi = x
        x = self.apply(
            inputs = x,
            name='%s-Norm' % timemix_name
        )
        
        arguments = {'with_state':True,'initial_state':False,
                     'input_mask':False,'rnn_mode':rnn_mode}
        inputs = [x]
        if mask is not None:
            arguments['input_mask']=True
            inputs.append(mask)
        if initial_state is not None:
            arguments['initial_state']=True
            inputs.append(initial_state)
        if rnn_mode:
            assert last_x is not None
            inputs.append(last_x[0])
        now_time_x = self.take_last_x(x,last_x_index)
        
        out = self.apply(
            inputs = inputs,
            arguments=arguments,
            name = timemix_name
        )
        x = out[0]
        x = self.apply(
            inputs=[xi, x], layer=keras.layers.Add, name='%s-Add' % timemix_name
        )

        xi = x

        x = self.apply(
            inputs = x,
            name='%s-Norm' % channelmix_name
        )
        now_channal_x = self.take_last_x(x,last_x_index)
        if rnn_mode:
            x = [x,last_x[1]]
        x = self.apply(
            inputs = x,
            name = channelmix_name,
            arguments={'rnn_mode':rnn_mode}
        )
        x = self.apply(
            inputs=[xi, x], layer=keras.layers.Add, name='%s-Add' % channelmix_name
        )
        return x,[now_time_x,now_channal_x],out[1]
    def enable_state_tunig(self,time_shitf_tuning=False):
        for layer in self.layers.values():
            if isinstance(layer,TimeMix) or isinstance(layer,ChannelMix):
                layer.enable_state_tunig(time_shitf_tuning)
            elif not lora_model:
                layer.trainable = False
    def apply_final_layers(self, x):
        x = self.apply(
            inputs = x,
            layer = keras.layers.LayerNormalization,
            epsilon=1e-5,
            name='Out-LN'
        )
        if self.with_lm:
            lm_activation = 'softmax' if self.with_lm is True else self.with_lm
            x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.vocab_size,
                    activation=lm_activation,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='RWKV-LM'
                )
        return x
    def set_outputs(self, outputs):
        super().set_outputs(outputs)
        if self.output_state:
            self.outputs.append(ops.stack(self.states,1))
    def slice_inputs(self,inputs,key,index):
        return ops.expand_dims(ops.take(inputs[key],index,axis=1),1)
    def get_prefill_mask(self,x):
        def mask(s):
            #mask要右移一位
            mask = ops.not_equal(s,0)
            mask = ops.pad(mask,[[0,0],[0,1]])
            return mask[:,1:]

        attention_bias = self.apply(
            inputs=x,
            layer=keras.layers.Lambda,
            function=mask,
            name='RKKV-Prefill-Mask'
        )

        return attention_bias
    def cache_call(self,inputs:list,input_lengths:list,end_token,
                   search_mode='greedy',k=1,progress_print=True,index_bias=0):
        old_flag = self.custom_position_ids
        
        x = inputs[0]
        
        if self.input_state:
            rnn_mode = True
            states = list(ops.unstack(inputs[1],self.num_hidden_layers,axis=1))
            last_xs = list(ops.unstack(inputs[2],self.num_hidden_layers*2,axis=1))
        else:
            rnn_mode = False
            states = [None]*self.num_hidden_layers
            last_xs = [None]*self.num_hidden_layers*2
        key = 0
        class start_index(keras.Layer):
            def call(self,x):
                z = x!=0
                if index_bias>0:
                    t = ops.ones([ops.shape(z)[0],index_bias],dtype=z.dtype)
                    z = ops.slice_update(z,[0,0],t)
                return ops.cast(ops.max(ops.sum(z,-1))-1,'int32')
        index = self.apply(
            inputs=x,
            layer=start_index,
            dtype='float32',
            name='start_index'
            
        )
        class Slices(keras.Layer):
            def call(self,x,index):
                #return x
                if backlib=='jax' or backlib=='tensorflow':
                    #jax不好编译，prefill要完全计算
                    return x
                ids = ops.arange(start=0,stop=index+1)
                z = ops.take(x,ids,axis=1)
                return z

            def compute_output_shape(self, input_shape):
                return input_shape
        z = Slices()(x,index)
        
        mask = self.get_prefill_mask(z)
        prefill_flag = ops.all(ops.not_equal(index,0))
        if keras.utils.is_keras_tensor(prefill_flag):
            prefill_flag = True
        if not self.input_state or prefill_flag:
            z = self.apply_embeddings([z])
            if backlib=='jax' or backlib=='tensorflow': 
                z *= ops.cast(mask[...,None],z.dtype)
            for i in range(self.num_hidden_layers):
                initial_states = states[i]
                last_x = last_xs[i*2:i*2+2]
                out=self.apply_main_cache_layers(z,i,initial_state=initial_states,
                                                mask=mask,rnn_mode=rnn_mode,last_x=last_x,last_x_index=index)

                z = out[0]
                last_xs[i*2:i*2+2]=out[1][:]
                states[i] = out[2]
        length = input_lengths[0]
        
        def cond(inputs, states, index,last_xs , flags):
            
            cond1 = ops.less(index,length-1)
            cond2 = ops.logical_not(ops.all(ops.equal(inputs[key][:,index],end_token),-1))
            return ops.logical_and(cond1,cond2)
        
        def body(inputs, states, index,last_xs , flags):
            if progress_print:
                print('\r',index,end='')
            

            xs = self.slice_inputs(inputs,key,index)
            '''print('\n')
            print(xs)
            print(inputs)'''
            z = self.apply_embeddings([xs])
            for i in range(self.num_hidden_layers):
                initial_states = states[i]
                last_x = last_xs[i*2:i*2+2]
                out=self.apply_main_cache_layers(z,i,initial_state=initial_states,
                                             mask=flags,rnn_mode=True,last_x=last_x)

                z = out[0]
                #last_xs[i*2:i*2+2]=out[1][:]
                last_xs[i*2] = ops.where(ops.expand_dims(flags,-1),out[1][0],last_xs[i*2])
                last_xs[i*2+1] = ops.where(ops.expand_dims(flags,-1),out[1][1],last_xs[i*2+1])
                states[i] = out[2]
            o = self.apply_final_layers(z)
            #print(o[:,:,:20])
            index += 1
            search_in = [o,index,inputs[key],flags]
            inputs[key],flags = self.Search(search_in,k=k,mode=search_mode)
            return (inputs, states, index,last_xs , flags)
        num_hidden_layers = self.num_hidden_layers
        self.end_token = end_token 
        class WhileLayer(keras.Layer):
            def __init__(self,wkv_dtype,**kwargs):
                super().__init__(**kwargs)
                self.wkv_dtype = wkv_dtype
            def call(self, x):
                inputs, states, index,last_xs =  x[:]
                xs = ops.expand_dims(ops.take(inputs[key],index,axis=1),1)
                flags = ops.not_equal(xs,end_token)
                #flags = ops.ones([ops.shape(inputs[key])[0],1],dtype='bool')
                for i in range(len(states)):
                    states[i] = ops.cast(states[i],self.wkv_dtype)
                
                if backlib=='torch':
                    while cond(inputs, states, index,last_xs , flags):
                        inputs, states, index,last_xs , flags = body(inputs, states, index,last_xs , flags)
                    return (inputs, states, index,last_xs)
                
                outs=ops.while_loop(
                    cond,
                    body,
                    loop_vars=(inputs, states, index,last_xs , flags),
                    maximum_iterations=length-index                                                                                                                                                                                                                                                                                                                                                       ,
                )
                if progress_print:
                    print('\n')
                return outs[:-1]
            def compute_output_shape(self, input_shape):
                return input_shape
            def get_config(self):
                config = {
                    'wkv_dtype': self.wkv_dtype,
                }
                base_config = super().get_config()
                return dict(list(base_config.items()) + list(config.items()))
        inputs = [x]
        out=self.apply(
            inputs=(inputs, states, index,last_xs),
            layer=WhileLayer,
            name='WhileLayer',
            wkv_dtype=self.wkv_dtype,
        )
        out_ids = ops.cast(out[0][key],'int32')
        if self.output_state:
            return [out_ids,ops.stack(out[1],axis=1),ops.stack(last_xs,axis=1)]
        return out_ids
    def get_cache_inputs(self,lengths:list):
        x_in = self.apply(
            layer=keras.Input, shape=[lengths[0],], name='Input-Token-cache-'+str(lengths[0]),dtype='int32',
        )
        inputs = [x_in]
        if self.input_state:
            inputs.append(
                self.apply(
                    layer=keras.Input, shape=(self.num_hidden_layers,
                        self.hidden_size // self.attention_head_size,
                        self.attention_head_size,self.attention_head_size), 
                        name='Stack-Statess',dtype=self.wkv_dtype

                )
            )

            inputs.append(
                self.apply(
                    layer=keras.Input, shape=(
                        self.num_hidden_layers*2,
                        1,
                        self.hidden_size), 
                        name='Stack-Last-X'
                )
            )
        return inputs
    def build_cache_model(self, input_lengths: list, end_token, search_mode='greedy', k=1, progress_print=False, index_bias=0,
                          input_state=False,output_state=False):
        olds = [self.input_state,self.output_state]
        self.input_state,self.output_state = input_state,output_state
        model = super().build_cache_model(input_lengths, end_token, search_mode, k, progress_print, index_bias)
        self.input_state,self.output_state = olds[:]
        return model
