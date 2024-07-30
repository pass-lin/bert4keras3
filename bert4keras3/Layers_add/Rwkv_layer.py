from bert4keras3.backend import ops,keras
from bert4keras3.backend import Layer,backlib
from keras.layers import Dense
from bert4keras3.Layers_add.LayerNorms import GroupNorm
def relu_square(x):
    return ops.square(ops.relu(x))

class DecomposerDense(Layer):
    def __init__(self,hidden_size,decomposer_size,use_bias=False,name="decomposed_dense"):
        super(DecomposerDense,self).__init__(name=name)
        self.hidden_size = hidden_size
        self.decomposer_size = decomposer_size
        self.use_bias = use_bias
    def build(self, input_shape):
        super().build(input_shape)
        self.dense_a = Dense(self.decomposer_size,activation='tanh',use_bias=False,name="dense_a")
        self.dense_b = Dense(self.hidden_size,use_bias=self.use_bias,name="dense_b")
    def call(self,inputs):
        x = self.dense_a(inputs)
        o = self.dense_b(x)
        return o
    def get_config(self):
        config = {
            'hidden_size' : self.hidden_size,
            'decomposer_size' : self.decomposer_size,
            'use_bias' : self.use_bias
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TimeShift(Layer):
    def __init__(self,name="time_shift"):
        super(TimeShift, self).__init__(name=name)
    def call(self, inputs,cache_x=None):
        x = ops.pad(inputs,[[0,0],[1,0],[0,0]],constant_values=0.)[:,:-1,:]
        if cache_x is not None:
            x = ops.slice_update(x,[0,0,0],cache_x)
        o = x - inputs
        return o
    def compute_output_shape(self, input_shape):
        return input_shape

class ChannelMix(Layer):
    def __init__(self,hidden_size,expand_size,**kwargs):
        super(ChannelMix, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.expand_size = expand_size
        self.supports_masking = True
        self.state_tuning = False
        self.time_shitf_tuning = False
    def call(self, inputs,rnn_mode = False):
        if rnn_mode:
            x = inputs[0]
            xx = self.timeshift(x,inputs[1])
        elif self.state_tuning and self.time_shitf_tuning:
            x = inputs
            xx = self.timeshift(inputs,self.last_x)
        else:
            x = inputs
            xx = self.timeshift(inputs)
        xk = x + xx * ops.reshape(self.time_mix_k,(1,1,self.hidden_size))
        xr = x + xx * ops.reshape(self.time_mix_r,(1,1,self.hidden_size))

        k = self.dense_key(xk)
        r = self.dense_receptance(xr)
        kv = self.dense_value(k)
        o = r * kv
        return o
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape,list):
            return input_shape[0]
        return input_shape
    def build(self, input_shape):
        super().build(input_shape)
        self.time_mix_k  = self.add_weight(shape=(self.hidden_size,),name="time_mix_k")
        self.time_mix_r  = self.add_weight(shape=(self.hidden_size,),name="time_mix_r")
        self.timeshift = TimeShift()
        self.dense_key = Dense(self.expand_size,activation=relu_square,use_bias=False,name="dense_k")
        self.dense_value = Dense(self.hidden_size,use_bias=False,name="dense_v")
        self.dense_receptance = Dense(self.hidden_size,activation=ops.sigmoid,use_bias=False,name="dense_r")
        self.hidden_size = self.hidden_size
        
    def enable_state_tunig(self,time_shitf_tuning=False):
        if self.state_tuning:
            raise ValueError(
                "state_tuning is already enabled. "
                "This can only be done once per layer."
            )
        self.time_shitf_tuning = time_shitf_tuning
        if time_shitf_tuning:
            self._tracker.unlock()
            self.last_x = self.add_weight(
                name="lora_last_x",
                shape=(1,1,self.hidden_size),
                initializer='zeros',dtype=self.wkv_dtype
            )
            self.time_mix_r.trainable = False
            self.time_mix_k.trainable = False
            self.dense_key.trainable = False
            self.dense_receptance.trainable = False
            self.dense_value.trainable = False
            
            self._tracker.lock()
            self.state_tuning = True
    def get_config(self):
        config = {
            'hidden_size':self.hidden_size,
            'expand_size':self.expand_size
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeMix(Layer):
    def __init__(self,layer_idx,rwkv_kernel,hidden_size,decomposer_size,head_size,time_decay_size=64,**kwargs):
        super(TimeMix, self).__init__(**kwargs)
        assert head_size % 4 ==0
        assert head_size % head_size == 0
        num_heads = hidden_size // head_size    
        self.time_shitf_tuning = False
        self.layer_idx = layer_idx
        self.head_size= head_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.supports_masking = True
        self.rwkv_kernel = rwkv_kernel
        self.decomposer_size = decomposer_size
        self.state_tuning = False
        self.time_decay_size = time_decay_size
        self.wkv_dtype = 'float32'
            
    def call(self, inputs,initial_state=False,input_mask=True,with_state=False,rnn_mode = False):
        x,n = inputs[0],1
        
        if input_mask:
            mask  = inputs[n]
            n += 1
        else:
            mask = None
        if initial_state:
            initial_state = ops.cast(inputs[n],self.wkv_dtype)
            n += 1
        elif self.state_tuning:
            #initial_state = self.initial_state
            initial_state = ops.cast(self.initial_state,self.wkv_dtype)
        else:
            initial_state = None
        
        if rnn_mode:
            x_shift = self.timeshift(x,inputs[n])
        elif self.state_tuning and self.time_shitf_tuning:
            x_shift = self.timeshift(x,self.last_x)
        else:
            x_shift = self.timeshift(x)
        
        x_mix = x + x_shift * ops.reshape(self.time_mix_x,(1, 1, self.hidden_size))
        
        
            #print(x_mix[:,:,:4])
        xr,xk,xv,xw,xg = self.dense_xr(x_mix),self.dense_xk(x_mix),self.dense_xv(x_mix),self.dense_xw(x_mix),self.dense_xg(x_mix)
        xr = x + x_shift * xr
        xk = x + x_shift * xk
        xv = x + x_shift * xv
        xw = x + x_shift * xw
        xg = x + x_shift * xg
        
        r,k,v,w,g = self.dense_r(xr),self.dense_k(xk),self.dense_v(xv),self.dense_w(xw),self.dense_g(xg)
        #print(mask)
        
        if mask!=None:
            k = ops.where(mask[...,None],k,0)
            v = ops.where(mask[...,None],v,0)
            w = ops.where(mask[...,None],w,-1e9)
        r,w,k,v,u = ops.cast(r,self.wkv_dtype),ops.cast(w,self.wkv_dtype),ops.cast(k,self.wkv_dtype),ops.cast(v,self.wkv_dtype),ops.cast(self.time_faaaa,self.wkv_dtype)
        x,output_state = self.rwkv_kernel(r,k,v,w,u,with_state=with_state,init_state=initial_state)
        #if output_state is not None:
        #    output_state = ops.cast(output_state,self.compute_dtype)
        x = self.group_norm(x)
        
        o = ops.cast(x,self.compute_dtype) * g

        o = self.dense_o(o)
        return [o,output_state]
    def build(self, input_shape):
        super().build(input_shape)
        self.timeshift = TimeShift(name="time_shift")
        self.dense_xr = DecomposerDense(self.hidden_size, self.decomposer_size,use_bias=True,name="decomposed_dense_xr")
        self.dense_xw = DecomposerDense(self.hidden_size, self.decomposer_size,use_bias=True,name="decomposed_dense_xw")
        self.dense_xk = DecomposerDense(self.hidden_size, self.decomposer_size,use_bias=True,name="decomposed_dense_xk")
        self.dense_xv = DecomposerDense(self.hidden_size, self.decomposer_size,use_bias=True,name="decomposed_dense_xv")
        self.dense_xg = DecomposerDense(self.hidden_size, self.decomposer_size,use_bias=True,name="decomposed_dense_xg")
        
        self.time_mix_x  = self.add_weight(shape=(self.hidden_size,),name="time_mix_x")

        self.dense_r = Dense(self.hidden_size,use_bias=False,name="dense_r")
        self.dense_k = Dense(self.hidden_size,use_bias=False,name="dense_k")
        self.dense_v = Dense(self.hidden_size,use_bias=False,name="dense_v")
        self.dense_w = DecomposerDense(self.hidden_size, self.time_decay_size,use_bias=True,name="decomposed_dense_w")
        self.dense_g = Dense(self.hidden_size,activation=ops.silu,use_bias=False,name="dense_g")
        
        self.time_faaaa = self.add_weight(shape=(self.num_heads,self.head_size),name="time_faaaa")
        
        self.group_norm = GroupNorm(self.hidden_size,self.head_size,name="group_ln",dtype=self.wkv_dtype)
        self.dense_o = Dense(self.hidden_size,use_bias=False,name="dense_o")
        self.initial_state = None
    def enable_state_tunig(self,time_shitf_tuning=False):
        if self.state_tuning:
            raise ValueError(
                "state_tuning is already enabled. "
                "This can only be done once per layer."
            )
        self._tracker.unlock()
        self.initial_state = self.add_weight(
            name="lora_kernel_a",
            shape=(1,self.hidden_size // self.head_size,self.head_size,self.head_size),
            initializer='zeros',dtype=self.wkv_dtype
        )
        if time_shitf_tuning:
            self.last_x = self.add_weight(
                name="lora_last_x",
                shape=(1,1,self.hidden_size),
                initializer='zeros',dtype=self.wkv_dtype
            )
        self.time_shitf_tuning = time_shitf_tuning
        self.dense_xr.trainable = False
        self.dense_xw.trainable = False
        self.dense_xk.trainable = False
        self.dense_xv.trainable = False
        self.dense_xg.trainable = False
        
        self.time_mix_x.trainable  = False

        self.dense_r.trainable = False
        self.dense_k.trainable = False
        self.dense_v.trainable = False
        self.dense_w.trainable = False
        self.dense_g.trainable = False
        
        self.time_faaaa.trainable = False
        
        self.group_norm.trainable = False
        self.dense_o.trainable = False
        self.initial_state.trainable = True
        self._tracker.lock()
        self.state_tuning = True
    def get_config(self):
        config = {
           'layer_idx': self.layer_idx , 
           'head_size' : self.head_size,
           'hidden_size':self.hidden_size,
           'num_heads':self.num_heads,
           'rwkv_kernel':self.rwkv_kernel,
           'time_decay_size':self.time_decay_size
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        B,T,C = input_shape[0]
        H = C // self.head_size
        return [input_shape[0],[B,H,self.head_size,self.head_size]]