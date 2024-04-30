


from models import *
tokenizer = AutoTokenizer.from_pretrained(dict_path)
class Llama_AE(Llama_AE_base):
    def __init__(self, **kwargs):
        super(Llama_AE, self).__init__(**kwargs)
        self.num_hidden_layers *=2 



Decoder = self = build_transformer_model(
    config_path,
    keras_weights_path=weights_path, 
    model=Llama_AE,
    with_lm=True,
    sequence_length=max_len,
    return_keras_model=False,
)


from bert4keras3.layers import Loss
class CrossEntropy(Loss):
    def compute_loss(self, inputs, mask=None):
        
        if len(inputs)==3:
            y_true, y_pred,loss = inputs
            loss = loss[:,:-1]
        else:
            y_true, y_pred = inputs
            loss = 0
        
        y_pred = ops.cast(y_pred,'float32')
        y_true = ops.cast(y_true,'float32')
        y_mask = ops.cast(ops.not_equal(y_true, 0), y_pred.dtype)
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        
        #loss2 = loss2[:,:-1]
        loss += ops.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
        loss = ops.sum(loss * y_mask,-1) / ops.sum(y_mask,-1)
        
        return ops.mean(loss)
    

decoder = Decoder.model

decoder.load_weights(initial_name)
out = CrossEntropy(1,dtype='float32')([decoder.inputs[0],
                                       decoder.outputs[0],
                                       Decoder.loss])


train_model = keras.Model(decoder.inputs,out)



train_model.get_layer('AutoEncoder').trainable = True
generate_model=Decoder.build_cache_model([max_len],end_token=end_token,search_mode='topp',k=0.6)

from copy import deepcopy
def predict(x):
    seqlen = x.shape[1]
    
    x = sequence_padding(deepcopy(x),length=max_len)
    start_prompt = tokenizer.encode('Step 1:')
    x[:,seqlen:seqlen+len(start_prompt)+1] = [start_token]+start_prompt
    out = generate_model.predict(x,batch_size=batch_size,verbose=1)
    return out[:,seqlen:] 
print(train_model.get_layer('AutoEncoder').weights)