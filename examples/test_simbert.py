# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:36:54 2024

@author: Administrator
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

from bert4keras3.backend import keras, K
from bert4keras3.models import *
from bert4keras3.tokenizers import Tokenizer
from bert4keras3.snippets import sequence_padding
import numpy as np

base_path = 'D:\ea下载\chinese_simbert_L-12_H-768_A-12/'
config_path = base_path+'bert_config.json'
checkpoint_path = base_path+'bert_model.ckpt'
dict_path = base_path+'vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
end_token=tokenizer._token_end_id
# 建立加载模型
self = bert = build_transformer_model(
    config_path,
    checkpoint_path, 
    model='bert',
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

#如果要测试检索，使用下面的代码

import numpy as np
import os

os.environ["KERAS_BACKEND"] = 'torch'


logits_base_path ='Roformer-sim-ft-Small/'
logits_config_path = logits_base_path+'bert_config.json'
logits_weights_path = logits_base_path+'model.weights.h5'
logits_dict_path = logits_base_path+'vocab.txt'

import keras
from bert4keras3.models import build_transformer_model
from bert4keras3.tokenizers import Tokenizer
from bert4keras3.snippets import sequence_padding

maxlen = 256
dtype = keras.config.dtype_policy()
keras.config.set_dtype_policy('float32')


bert = build_transformer_model(
        config_path=logits_config_path,
        keras_weights=logits_weights_path,
        model='roformer',
        return_keras_model=False,
        with_pool='linear',
        with_mlm=False,
    )

tokenizer = Tokenizer(logits_dict_path, do_lower_case=True)  # 建立分词器
encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=-1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def similarity(text1, text2):
    """"计算text1与text2的相似度
    """
    texts = [text1, text2]
    X, S = [], []
    for t in texts:
        x, s = tokenizer.encode(t, maxlen=maxlen)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = l2_normalize(encoder.predict([X, S],verbose=3))
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    print((Z[0] * Z[1]).sum())


similarity(u'今天天气不错', u'今天天气很好')
similarity(u'今天天气不错', u'今天天气不好')
similarity(u'我喜欢北京', u'我很喜欢北京')
similarity(u'我喜欢北京', u'我不喜欢北京')
similarity(u'电影不错', u'电影很好')
similarity(u'电影不错', u'电影不好')
similarity(u'红色的苹果', u'绿色的苹果')
similarity(u'给我推荐一款红色的车', u'给我推荐一款黑色的车')
similarity(u'给我推荐一款红色的车', u'推荐一辆红车')
similarity(u'给我推荐一款红色的车', u'麻烦来一辆红车')
