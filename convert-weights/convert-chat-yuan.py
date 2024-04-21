# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:52:38 2024

@author: Administrator
"""
import os

#os.environ["KERAS_BACKEND"] = "jax"
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
import keras

from bert4keras3.models import build_transformer_model as build_transformer_model_keras
print('loading keras model')
t5 = build_transformer_model_keras(
    config_path='ChatYuan-keras/config.json',
    model='mt5.1.1',
    return_keras_model=False,
    with_lm='linear',
    name='T5',
)

tfencoder = t5.encoder
tfdecoder = t5.decoder
tfmodel = t5.model


# 配置
pretrain_model = 'ChatYuan-torch/'
config_path = pretrain_model + 'config.json'
checkpoint_path = pretrain_model + 'pytorch_model.bin'
spm_path = pretrain_model + 'spiece.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>', keep_accents=True)
print('loading torch model')
encoder = build_transformer_model(config_path, checkpoint_path, pad_token_id=-1).to(device)

print('finish loading')

weights=encoder.state_dict()
def get_weight(key):
    weight=weights[key].cpu().numpy()
    
    return weight
embeddingweight = weights['encoder.embeddings.word_embeddings.weight'].cpu().numpy()
enocder_embeding = tfmodel.get_layer('Embedding-Token')
enocder_embeding.set_weights([embeddingweight])
del(weights['encoder.embeddings.word_embeddings.weight'])
import gc
for index in range(24):
    get_weight('encoder.encoderLayer.%d.multiHeadAttention.relative_positions.relative_position'% index)
    get_weight('decoder.decoderLayer.%d.multiHeadAttention.relative_positions.relative_position'% index)
    encoder_relative = get_weight('encoder.encoderLayer.%d.multiHeadAttention.relative_positions_encoding.weight'% index)
    decoder_relative = get_weight('decoder.decoderLayer.%d.multiHeadAttention.relative_positions_encoding.weight'% index)
    
    attention_name = 'Encoder-Transformer-%d-MultiHeadSelfAttention' % index
    feed_forward_name = 'Encoder-Transformer-%d-FeedForward' % index
    att_ln =tfmodel.get_layer('%s-Norm' % attention_name)
    att = tfmodel.get_layer(attention_name)
    ffn_ln = tfmodel.get_layer('%s-Norm' % feed_forward_name)
    ffn = tfmodel.get_layer(feed_forward_name)
    
    q_weight = get_weight('encoder.encoderLayer.%d.multiHeadAttention.q.weight'% index)
    k_weight = get_weight('encoder.encoderLayer.%d.multiHeadAttention.k.weight'% index)
    v_weight = get_weight('encoder.encoderLayer.%d.multiHeadAttention.v.weight'% index)
    o_weight = get_weight('encoder.encoderLayer.%d.multiHeadAttention.o.weight'% index)
    
    att.set_weights([q_weight.T,k_weight.T,v_weight.T,o_weight.T])
    
    weight = get_weight('encoder.encoderLayer.%d.attnLayerNorm.weight'% index)
    att_ln.set_weights([weight])
    
    w1 = get_weight('encoder.encoderLayer.%d.feedForward.intermediateDense.weight'% index)
    w2 = get_weight('encoder.encoderLayer.%d.feedForward.intermediateDense1.weight'% index)
    w3 = get_weight('encoder.encoderLayer.%d.feedForward.outputDense.weight'% index)
    
    ffn.set_weights([w1.T,w2.T,w3.T])
    

    w=get_weight('encoder.encoderLayer.%d.ffnLayerNorm.weight'% index)
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
    
    
    w=get_weight('decoder.decoderLayer.%d.crossLayerNorm.weight'% index)
    crossatt_ln.set_weights([w])
    
    q_weight = get_weight('decoder.decoderLayer.%d.crossAttention.q.weight'% index)
    k_weight = get_weight('decoder.decoderLayer.%d.crossAttention.k.weight'% index)
    v_weight = get_weight('decoder.decoderLayer.%d.crossAttention.v.weight'% index)
    o_weight = get_weight('decoder.decoderLayer.%d.crossAttention.o.weight'% index)
    
    crossatt.set_weights([q_weight.T,k_weight.T,v_weight.T,o_weight.T])
    
    w=get_weight('decoder.decoderLayer.%d.attnLayerNorm.weight'% index)
    selfatt_ln.set_weights([w])
    
    q_weight = get_weight('decoder.decoderLayer.%d.multiHeadAttention.q.weight'% index)
    k_weight = get_weight('decoder.decoderLayer.%d.multiHeadAttention.k.weight'% index)
    v_weight = get_weight('decoder.decoderLayer.%d.multiHeadAttention.v.weight'% index)
    o_weight = get_weight('decoder.decoderLayer.%d.multiHeadAttention.o.weight'% index)
    
    selfatt.set_weights([q_weight.T,k_weight.T,v_weight.T,o_weight.T])
    
    w1 = get_weight('decoder.decoderLayer.%d.feedForward.intermediateDense.weight'% index)
    w2 = get_weight('decoder.decoderLayer.%d.feedForward.intermediateDense1.weight'% index)
    w3 = get_weight('decoder.decoderLayer.%d.feedForward.outputDense.weight'% index)
    
    ffn.set_weights([w1.T,w2.T,w3.T])
    

    w=get_weight('decoder.decoderLayer.%d.ffnLayerNorm.weight'% index)
    ffn_ln.set_weights([w])
    gc.collect()

encoder_p=tfmodel.get_layer('Encoder-Embedding-Relative-Position')
decoder_p=tfmodel.get_layer('Decoder-Embedding-Relative-Position')
encoder_p.set_weights([encoder_relative])
decoder_p.set_weights([decoder_relative])

w=get_weight('encoder.final_layer_norm.weight')
tfmodel.get_layer('Encoder-Output-Norm').set_weights([w])

decoder_em=get_weight('decoder.embeddings.word_embeddings.weight')

w=get_weight('decoder.lm_head.weight')
tfmodel.get_layer('Decoder-Output-LM').set_weights([w.T])

w=get_weight('decoder.final_layer_norm.weight')
tfmodel.get_layer('Decoder-Output-Norm').set_weights([w])

#tfmodel.save_weights('ChatYuan-keras/T5weights.weights.h5')
#test
print('test')
import numpy as np
text='我'
token_ids, _ = tokenizer.encode(text, maxlen=768)
token_ids = torch.tensor([token_ids], device=device)
encoder_output = encoder.encoder.predict([token_ids])
z=encoder.decoder.predict([token_ids]+encoder_output )

x=tokenizer.encode(text, maxlen=768)[0]
x=np.reshape(x,[1,2])
e1=tfencoder.predict(x)
z2=tfdecoder.predict([e1,x])
