# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:09:53 2023

@author: Administrator
"""

#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征
# 权重链接: https://pan.baidu.com/s/1-FB0yl1uxYDCGIRvU1XNzQ 提取码: xynn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
from bert4keras3.backend import keras
#from bert4keras3.backend import ops
from bert4keras3.models import build_transformer_model
from bert4keras3.tokenizers import Tokenizer
from bert4keras3.snippets import to_array
#bert from 
base_path='models/chinese_t5_pegasus_base/'
config_path =base_path+ 'config.json'
checkpoint_path = base_path+'model.ckpt'
dict_path = base_path+'vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path,model='mt5.1.1_encoder',)  # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
token_ids, segment_ids = to_array([token_ids], [segment_ids])

print('\n ===== predicting =====\n')
print(model.predict([token_ids]))
model.predict([np.zeros([1024,32])],batch_size=8)
model.predict([np.zeros([1024,32])],batch_size=8,verbose=1)

"""
cpu is  i7-9750H CPU 
gpu is 16660ti

keras2.3.1 with tf2.2 ：
这里要乘8才和后面的等价
cpu:
1024/1024 [==============================] - 29s 28ms/step
[[[ 2.9163269e-02 -2.3002862e-03 -1.7774916e-03 ...  6.4450777e-03
    1.8854957e-03 -3.2775817e-04]
  [-1.0686934e-01  4.8507418e-02  3.7731823e-02 ... -2.1010849e-01
   -1.4622805e-01  1.2896894e-01]
  [-1.5603334e-01 -2.8059089e-01  1.9916685e-01 ... -1.6994826e-02
   -2.9859442e-01  4.9437497e-02]
  [-1.0041825e-01  7.8659192e-02 -1.3222913e-01 ...  9.3017802e-02
   -3.9462018e-01  9.1594376e-02]
  [ 8.8463195e-02 -1.5324140e-01  1.5596795e-02 ...  1.8167461e-01
   -7.8392312e-02 -1.6704443e-01]
  [ 3.0728497e-02  6.5759005e-04 -3.2774259e-03 ...  5.3948797e-03
    2.3090802e-03 -5.0980033e-04]]]

keras3 torch backend
cpu:
128/128 ━━━━━━━━━━━━━━━━━━━━ 39s 306ms/step 
[[[ 2.91632637e-02 -2.30028783e-03 -1.77749514e-03 ...  6.44507399e-03
    1.88550167e-03 -3.27756570e-04]
  [-1.06869422e-01  4.85075675e-02  3.77318412e-02 ... -2.10108310e-01
   -1.46227852e-01  1.28969222e-01]
  [-1.56033427e-01 -2.80590594e-01  1.99166834e-01 ... -1.69949774e-02
   -2.98594326e-01  4.94375639e-02]
  [-1.00418195e-01  7.86591992e-02 -1.32229090e-01 ...  9.30177569e-02
   -3.94620210e-01  9.15944129e-02]
  [ 8.84632617e-02 -1.53241545e-01  1.55966021e-02 ...  1.81674406e-01
   -7.83924609e-02 -1.67044356e-01]
  [ 3.07285022e-02  6.57587079e-04 -3.27742612e-03 ...  5.39487787e-03
    2.30908417e-03 -5.09799924e-04]]]

keras3 jax backend
cpu 
128/128 ━━━━━━━━━━━━━━━━━━━━ 47s 365ms/step

[[[ 2.91632731e-02 -2.30029435e-03 -1.77749281e-03 ...  6.44507166e-03
    1.88549806e-03 -3.27762944e-04]
  [-1.06869273e-01  4.85074781e-02  3.77319790e-02 ... -2.10108250e-01
   -1.46227807e-01  1.28969267e-01]
  [-1.56033128e-01 -2.80590683e-01  1.99166790e-01 ... -1.69948954e-02
   -2.98594326e-01  4.94374521e-02]
  [-1.00418165e-01  7.86591992e-02 -1.32229015e-01 ...  9.30178016e-02
   -3.94620210e-01  9.15944204e-02]
  [ 8.84632021e-02 -1.53241694e-01  1.55966645e-02 ...  1.81674674e-01
   -7.83923790e-02 -1.67044431e-01]
  [ 3.07284966e-02  6.57585682e-04 -3.27742542e-03 ...  5.39487973e-03
    2.30908650e-03 -5.09798469e-04]]]


keras3 tf backend
cpu
128/128 ━━━━━━━━━━━━━━━━━━━━ 27s 196ms/step
[[[ 2.91632675e-02 -2.30028899e-03 -1.77749502e-03 ...  6.44507585e-03
    1.88549876e-03 -3.27757705e-04]
  [-1.06869288e-01  4.85075675e-02  3.77318524e-02 ... -2.10108206e-01
   -1.46227896e-01  1.28969163e-01]
  [-1.56033218e-01 -2.80590683e-01  1.99166775e-01 ... -1.69948991e-02
   -2.98594445e-01  4.94375080e-02]
  [-1.00418255e-01  7.86592141e-02 -1.32228926e-01 ...  9.30176899e-02
   -3.94620180e-01  9.15943086e-02]
  [ 8.84633809e-02 -1.53241530e-01  1.55967148e-02 ...  1.81674451e-01
   -7.83923939e-02 -1.67044491e-01]
  [ 3.07285022e-02  6.57586381e-04 -3.27742775e-03 ...  5.39488299e-03
    2.30908347e-03 -5.09799633e-04]]]
"""
