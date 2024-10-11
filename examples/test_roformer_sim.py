# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 23:14:43 2024

@author: Administrator
"""

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