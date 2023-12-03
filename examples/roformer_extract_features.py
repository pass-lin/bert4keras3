# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:09:53 2023

@author: Administrator
"""

#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "jax"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
from bert4keras3.backend import keras
#from bert4keras3.backend import ops
from bert4keras3.models import build_transformer_model
from bert4keras3.tokenizers import Tokenizer
from bert4keras3.snippets import to_array
#bert from 
config_path = 'models/chinese_roformer_L-6_H-384_A-6/bert_config.json'
checkpoint_path = 'models/chinese_roformer_L-6_H-384_A-6/bert_model.ckpt'
dict_path = 'models/chinese_roformer_L-6_H-384_A-6/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path,model='roformer')  # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
token_ids, segment_ids = to_array([token_ids], [segment_ids])

print('\n ===== predicting =====\n')
print(model.predict([token_ids, segment_ids]))
model.predict([np.zeros([1024,32]),np.zeros([1024,32])],batch_size=8)
model.predict([np.zeros([1024,32]),np.zeros([1024,32])],batch_size=8,verbose=1)

"""
cpu is  i7-9750H CPU 
gpu is 16660ti

keras2.3.1 with tf2.2 ：
这里要乘8才和后面的等价
gpu:
1024/1024 [==============================] - 1s 687us/step
cpu:
1024/1024 [==============================] - 7s 7ms/step
[[[-1.4107976  -0.5262189   0.1014159  ...  0.68842906  0.61351925
   -0.47100535]
  [-0.34521595 -0.071702    0.22511655 ...  0.62177175  0.06363515
   -0.16372496]
  [-1.0185404  -0.481669   -0.8060608  ... -0.28526458  0.7257574
    0.73347473]
  [ 0.08699846 -0.03402305  0.5098496  ...  0.85124403 -0.4309994
   -1.0549388 ]
  [-0.5224491   0.1698376  -0.36339054 ... -0.33867067  0.7168974
    0.4039372 ]
  [-1.537687   -0.20126376 -0.084051   ...  0.5513399   0.86524004
   -0.15430301]]]

keras3 torch backend
gpu:128/128 ━━━━━━━━━━━━━━━━━━━━ 6s 44ms/step 
cpu:128/128 ━━━━━━━━━━━━━━━━━━━━ 10s 78ms/step
[[[-1.4107975  -0.52621907  0.10141644 ...  0.68842965  0.61351764
   -0.47100517]
  [-0.3452167  -0.07170235  0.22511709 ...  0.62177265  0.06363471
   -0.16372572]
  [-1.0185413  -0.48166847 -0.80606115 ... -0.28526402  0.7257573
    0.7334743 ]
  [ 0.08699675 -0.03402366  0.50984985 ...  0.85124433 -0.43100032
   -1.0549383 ]
  [-0.5224489   0.16983745 -0.36339095 ... -0.3386695   0.716896
    0.40393758]
  [-1.5376867  -0.2012644  -0.0840507  ...  0.55134135  0.8652392
   -0.15430221]]]

jax 在windows cpu上比较慢，gpu上可以更快
keras3 jax backend
cpu 128/128 ━━━━━━━━━━━━━━━━━━━━ 9s 74ms/step
[[[-1.4107977  -0.526218    0.10141654 ...  0.6884298   0.61351854
   -0.47100553]
  [-0.34521535 -0.07170212  0.22511695 ...  0.621772    0.0636354
   -0.16372555]
  [-1.0185416  -0.4816675  -0.8060607  ... -0.28526476  0.72575724
    0.73347414]
  [ 0.08699801 -0.03402372  0.50985044 ...  0.851244   -0.43100035
   -1.054939  ]
  [-0.52244925  0.16983747 -0.36338997 ... -0.3386699   0.716897
    0.40393704]
  [-1.5376873  -0.20126374 -0.08405027 ...  0.5513404   0.8652398
   -0.15430279]]]

#tensorflow1.16有针对 windwos cpu的 优化，所以更快
keras3 tf backend
cpu 128/128 ━━━━━━━━━━━━━━━━━━━━ 7s 38ms/step
[[[-1.4107982  -0.5262194   0.1014161  ...  0.68842995  0.6135187
   -0.47100574]
  [-0.34521633 -0.07170162  0.2251176  ...  0.6217721   0.06363508
   -0.16372554]
  [-1.0185413  -0.48166904 -0.80606085 ... -0.285264    0.7257575
    0.73347425]
  [ 0.086998   -0.03402391  0.5098497  ...  0.8512442  -0.4310002
   -1.0549378 ]
  [-0.52244896  0.16983734 -0.36339077 ... -0.3386699   0.7168955
    0.40393674]
  [-1.5376865  -0.20126426 -0.08405063 ...  0.5513409   0.86523926
   -0.15430252]]]

"""

