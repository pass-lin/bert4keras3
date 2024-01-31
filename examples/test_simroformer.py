# -*- coding: utf-8 -*-
"""
jax torch tensorflow测试通过
理论上roformerV2也能用，但我没roformerV2的unilm模型，所以没测
"""
import os
os.environ["KERAS_BACKEND"] = "jax"

from bert4keras3.backend import keras, K
from bert4keras3.models import *
from bert4keras3.tokenizers import Tokenizer
from bert4keras3.snippets import sequence_padding
import numpy as np

base_path = 'D:\ea下载\chinese_roformer-sim-char_L-12_H-768_A-12/'
config_path = base_path+'bert_config.json'
checkpoint_path = base_path+'bert_model.ckpt'
dict_path = base_path+'vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
end_token=tokenizer._token_end_id
# 建立加载模型
self = bert = build_transformer_model(
    config_path,
    checkpoint_path, 
    model='roformer',
    application='unilm',
    with_mlm=True,
    return_keras_model=False,
)
#用没有cache的模型做greedy search
tokens,segments = tokenizer.encode('广东省的省会是广州')
l = len(tokens)
tokens = tokens+[tokenizer._token_start_id]
segments  = segments+[1]
#search
while tokens[-1]!=end_token:
    inputs = [np.expand_dims(tokens,0),np.expand_dims(segments,0)]
    pred=bert.model.predict(inputs,verbose=3)
    pred = pred.argmax(-1)[0][-1]
    tokens.append(pred)
    segments.append(1)
#展示结果
s2= segments
outs = tokens
print(tokenizer.decode(outs[l:]))


#cache模型做greedy saerch
max_len=32
input_lengths=[max_len,max_len]#segment和tokens的maxlen是一样
#构建输入
tokens,segments = tokenizer.encode('广东省的省会是广州')
tokens = np.expand_dims(tokens+[tokenizer._token_start_id] +[0]*(max_len-len(tokens)-1),0)
segments  = np.expand_dims(segments+[1]*(max_len-len(segments)),0)
inputs = [tokens,segments]
#构建cache模型
cache_model=bert.build_cache_model(input_lengths,end_token=end_token,progress_print=True)
#输出并展示结果
o1 = cache_model.predict([tokens,segments])
print(tokenizer.decode(o1[0][l:]))