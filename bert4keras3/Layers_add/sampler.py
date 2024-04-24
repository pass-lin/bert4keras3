from bert4keras3.backend import keras, ops , np
from keras import Layer
class TakeLayer(Layer):
    def __init__(self, axis=0, **kwargs):
        super(TakeLayer, self).__init__(**kwargs)
        self.axis = axis
    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(TakeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self,inputs, **kwargs):
        index = kwargs.get('index') 
        att = kwargs.get('att')
        out = ops.expand_dims(ops.take(inputs,index,self.axis),self.axis)
        return out
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[self.axis]=1
        return input_shape

class SearchBase(Layer):
    def __init__(self, end_token,k=1, **kwargs):
        super(SearchBase, self).__init__(**kwargs)
        self.k = k
        self.end_token=end_token
    def get_config(self):
        config = {
            'k': self.k,
            'end_token':self.end_token
        }
        base_config = super(SearchBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def sample(self,x):
        x = ops.cast(ops.log(x), "float32")
        
        return keras.random.categorical(
            # tf does not support half precision multinomial sampling, so make
            # sure we have full precision here.
            x,
            1,
            seed=np.random.randint(1,2147483648),
            dtype="int32",
        )
        
class GreedySearch(SearchBase):
    def search(self,t):
        return ops.argmax(t,-1)
    def call(self, inputs, **kwargs):
        hidden_state,update_index,out_ids,flags = inputs[:]
        
        y = self.search(hidden_state) 
        
        t = ops.full_like(y,self.end_token)
        y = ops.where(flags,y,t)
        start = [0,update_index]
        flags = y!=self.end_token

        return ops.slice_update(out_ids,start,ops.cast(y,out_ids.dtype)),flags
    
class TopkSearch(GreedySearch):
    def search(self,t):
        top_k_pred, top_k_indices = ops.top_k(
            t[:,0],
            k=self.k,
            sorted=False,
        )
        
        sample_indices = self.sample(top_k_pred)

        return ops.take_along_axis(top_k_indices, sample_indices, axis=-1)

class ToppSearch(TopkSearch):
    def search(self,t):
        cutoff = ops.shape(t)[-1]
        sorted_preds, sorted_indices = ops.top_k(
            t[:,0], k=cutoff, sorted=True
        )
        # Calculate cumulative probability distribution.
        cumulative_probabilities = ops.cumsum(sorted_preds, axis=-1)
        # Create a mask for the tokens to keep.
        keep_mask = cumulative_probabilities <= self.k
        # Shift to include the last token that exceed p.
        shifted_keep_mask = ops.concatenate(
            [ops.ones_like(keep_mask[:, :1]), keep_mask[:, :-1]], axis=1
        )
        # Filter out unmasked tokens and sample from filtered distribution.
        probabilities = ops.where(
            shifted_keep_mask,
            sorted_preds,
            ops.zeros(ops.shape(sorted_preds), dtype=sorted_preds.dtype),
        )
        sorted_next_token = self.sample(probabilities)
        output = ops.take_along_axis(sorted_indices, sorted_next_token, axis=-1)
        return output
