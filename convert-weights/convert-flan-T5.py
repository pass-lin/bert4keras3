# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:37:21 2024

@author: Administrator
"""
#转化transformers 的Flan T5
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:52:38 2024

@author: Administrator
"""
config={
 "hidden_dropout_prob": 0.1,
 "hidden_size": 1024,
 "initializer_range": 0.02,
 "intermediate_size": 2816,
 "num_attention_heads": 16,
 "attention_head_size": 64,
 "num_hidden_layers": 24,
 "vocab_size": 32128,
 "cross_position_bias": 0,
 "logit_scale": 0,
 "hidden_act": ["gelu", "linear"]
}

import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
from transformers import AutoConfig, T5ForConditionalGeneration,T5Model
from bert4keras3.models import build_transformer_model
model_name='flan-t5-small'
t5config =  AutoConfig.from_pretrained('google/'+model_name)
model = T5ForConditionalGeneration.from_pretrained('google/'+model_name)

config['hidden_dropout_prob'] = t5config.dropout_rate
config['hidden_size'] = t5config.d_model
config['intermediate_size'] = t5config.d_ff
config['num_attention_heads'] = t5config.num_heads
config['num_hidden_layers'] = t5config.num_layers
config['attention_head_size'] = t5config.d_kv
config['vocab_size'] = t5config.vocab_size
def get_weight(key):
    weight=weights[key].cpu().numpy()
    return weight

import json
with open(model_name+'/t5config.json', 'w') as f:
    json.dump(config, f)
    
kerasT5 = build_transformer_model(
        config_path=model_name+'/t5config.json',
        model='t5.1.1',
        return_keras_model=False,
        with_lm='linear',
    )

weights=model.state_dict()
def get_weight(key):
    weight=weights.pop(key).cpu().numpy()
    return weight

tfmodel = kerasT5.model
embeddingweight = get_weight('encoder.embed_tokens.weight')
enocder_embeding = tfmodel.get_layer('Embedding-Token')
enocder_embeding.set_weights([embeddingweight])
for index in range(t5config.num_layers):
    
    attention_name = 'Encoder-Transformer-%d-MultiHeadSelfAttention' % index
    feed_forward_name = 'Encoder-Transformer-%d-FeedForward' % index
    att_ln =tfmodel.get_layer('%s-Norm' % attention_name)
    att = tfmodel.get_layer(attention_name)
    ffn_ln = tfmodel.get_layer('%s-Norm' % feed_forward_name)
    ffn = tfmodel.get_layer(feed_forward_name)
    
    q_weight = get_weight('encoder.block.%d.layer.0.SelfAttention.q.weight'% index)
    k_weight = get_weight('encoder.block.%d.layer.0.SelfAttention.k.weight'% index)
    v_weight = get_weight('encoder.block.%d.layer.0.SelfAttention.v.weight'% index)
    o_weight = get_weight('encoder.block.%d.layer.0.SelfAttention.o.weight'% index)
    
    att.set_weights([q_weight.T,k_weight.T,v_weight.T,o_weight.T])
    weight = get_weight('encoder.block.%d.layer.0.layer_norm.weight'% index)
    att_ln.set_weights([weight])
    
    w1 = get_weight('encoder.block.%d.layer.1.DenseReluDense.wi_0.weight'% index)
    w2 = get_weight('encoder.block.%d.layer.1.DenseReluDense.wi_1.weight'% index)
    w3 = get_weight('encoder.block.%d.layer.1.DenseReluDense.wo.weight'% index)
    
    ffn.set_weights([w1.T,w2.T,w3.T])
    
    w=get_weight('encoder.block.%d.layer.1.layer_norm.weight'% index)
    ffn_ln.set_weights([w])

    self_attention_name = 'Decoder-Transformer-%d-MultiHeadSelfAttention' % index
    cross_attention_name = 'Decoder-Transformer-%d-MultiHeadCrossAttention' % index
    feed_forward_name = 'Decoder-Transformer-%d-FeedForward' % index
    
    selfatt_ln =tfmodel.get_layer('%s-Norm' % self_attention_name)
    selfatt = tfmodel.get_layer(self_attention_name)
        
    crossatt_ln =tfmodel.get_layer('%s-Norm' % cross_attention_name)
    crossatt = tfmodel.get_layer(cross_attention_name)
    
    ffn_ln = tfmodel.get_layer('%s-Norm' % feed_forward_name)
    ffn = tfmodel.get_layer(feed_forward_name)
    
    w=get_weight('decoder.block.%d.layer.0.layer_norm.weight'% index)
    selfatt_ln.set_weights([w])
    
    q_weight = get_weight('decoder.block.%d.layer.0.SelfAttention.q.weight'% index)
    k_weight = get_weight('decoder.block.%d.layer.0.SelfAttention.k.weight'% index)
    v_weight = get_weight('decoder.block.%d.layer.0.SelfAttention.v.weight'% index)
    o_weight = get_weight('decoder.block.%d.layer.0.SelfAttention.o.weight'% index)
    
    selfatt.set_weights([q_weight.T,k_weight.T,v_weight.T,o_weight.T])
    
    w=get_weight('decoder.block.%d.layer.1.layer_norm.weight'% index)
    crossatt_ln.set_weights([w])
    
    q_weight = get_weight('decoder.block.%d.layer.1.EncDecAttention.q.weight'% index)
    k_weight = get_weight('decoder.block.%d.layer.1.EncDecAttention.k.weight'% index)
    v_weight = get_weight('decoder.block.%d.layer.1.EncDecAttention.v.weight'% index)
    o_weight = get_weight('decoder.block.%d.layer.1.EncDecAttention.o.weight'% index)
    
    crossatt.set_weights([q_weight.T,k_weight.T,v_weight.T,o_weight.T])
    
    w1 = get_weight('decoder.block.%d.layer.2.DenseReluDense.wi_0.weight'% index)
    w2 = get_weight('decoder.block.%d.layer.2.DenseReluDense.wi_1.weight'% index)
    w3 = get_weight('decoder.block.%d.layer.2.DenseReluDense.wo.weight'% index)
    
    ffn.set_weights([w1.T,w2.T,w3.T])
    

    w=get_weight('decoder.block.%d.layer.2.layer_norm.weight'% index)
    ffn_ln.set_weights([w])
    
encoder_relative = get_weight('encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight')
decoder_relative = get_weight('decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight')
encoder_p=tfmodel.get_layer('Encoder-Embedding-Relative-Position')
decoder_p=tfmodel.get_layer('Decoder-Embedding-Relative-Position')
encoder_p.set_weights([encoder_relative])
decoder_p.set_weights([decoder_relative])


w=get_weight('encoder.final_layer_norm.weight')
tfmodel.get_layer('Encoder-Output-Norm').set_weights([w])

w=get_weight( 'decoder.final_layer_norm.weight')
tfmodel.get_layer('Decoder-Output-Norm').set_weights([w])

w=get_weight('lm_head.weight')
tfmodel.get_layer('Decoder-Output-LM').set_weights([w.T])



import numpy as np
import keras

x = np.reshape([[1,2,3,4,5,6,7,8,9,10]],[2,5])
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/'+model_name)
input_ids = tokenizer(
    "Studies have been shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

def printf():
    try:
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)['last_hidden_state']
    except:
        pass
    
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)['logits']
print(outputs)

keras_out = tfmodel.predict([input_ids,decoder_input_ids])
print(keras_out)
