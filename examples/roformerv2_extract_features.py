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
config_path = 'models/chinese_roformer-v2-char_L-6_H-384_A-6/bert_config.json'
checkpoint_path = 'models/chinese_roformer-v2-char_L-6_H-384_A-6/bert_model.ckpt'
dict_path = 'models/chinese_roformer-v2-char_L-6_H-384_A-6/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path,model='roformer_v2')  # 建立模型，加载权重

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
1024/1024 [==============================] - 1s 709us/step
cpu:
1024/1024 [==============================] - 6s 6ms/step
[[[-0.68473476 -1.1189076   0.8374482  ... -1.3070096  -0.99434096
   -0.8054465 ]
  [ 0.6301648  -0.8818993   0.6760466  ...  0.19107008 -0.4742949
   -0.36254817]
  [ 0.5119717  -1.0673342   0.7168075  ... -0.4195901  -1.1917425
   -0.6756266 ]
  [ 0.52944565 -1.2013384   0.3279568  ...  0.06762145 -0.7880495
   -1.1422358 ]
  [ 0.5265356  -1.7204537   0.78924567 ... -0.77193075 -1.1719868
   -1.1538503 ]
  [ 0.50315833 -0.8803192   0.05012124 ... -0.49182516 -0.322246
   -0.5626907 ]]]

keras3 torch backend
gpu:128/128 ━━━━━━━━━━━━━━━━━━━━ 6s 47ms/step  
cpu:128/128 ━━━━━━━━━━━━━━━━━━━━ 8s 66ms/step
[[[-0.68473524 -1.1189071   0.83744806 ... -1.3070102  -0.9943409
   -0.8054467 ]
  [ 0.63016415 -0.8818989   0.67604697 ...  0.19106974 -0.47429508
   -0.3625478 ]
  [ 0.5119719  -1.0673342   0.7168071  ... -0.41959107 -1.1917427
   -0.67562634]
  [ 0.52944505 -1.2013388   0.32795823 ...  0.0676218  -0.7880501
   -1.1422344 ]
  [ 0.5265348  -1.7204543   0.7892456  ... -0.7719308  -1.171986
   -1.1538494 ]
  [ 0.5031582  -0.8803184   0.05012038 ... -0.4918253  -0.32224643
   -0.56268954]]]

keras3 jax backend
cpu 128/128 ━━━━━━━━━━━━━━━━━━━━ 6s 50ms/step
[[[-0.6847347  -1.1189075   0.837448   ... -1.3070099  -0.9943406
   -0.80544716]
  [ 0.63016444 -0.8818989   0.6760474  ...  0.19107    -0.47429475
   -0.36254838]
  [ 0.5119723  -1.0673343   0.7168076  ... -0.4195907  -1.1917423
   -0.67562723]
  [ 0.5294451  -1.2013391   0.32795763 ...  0.06762175 -0.7880493
   -1.1422349 ]
  [ 0.52653486 -1.7204547   0.78924596 ... -0.7719309  -1.1719867
   -1.1538502 ]
  [ 0.5031577  -0.88031924  0.05012183 ... -0.49182656 -0.3222465
   -0.56269026]]]


keras3 tf backend
cpu 128/128 ━━━━━━━━━━━━━━━━━━━━ 4s 30ms/step
[[[-0.68473536 -1.1189077   0.8374477  ... -1.30701    -0.9943408
   -0.8054466 ]
  [ 0.6301638  -0.88189876  0.6760464  ...  0.19106938 -0.4742958
   -0.36254728]
  [ 0.5119716  -1.0673348   0.7168065  ... -0.41959095 -1.1917434
   -0.6756273 ]
  [ 0.52944416 -1.2013392   0.32795805 ...  0.06762107 -0.78805065
   -1.1422353 ]
  [ 0.52653414 -1.7204542   0.78924537 ... -0.77193123 -1.1719865
   -1.1538494 ]
  [ 0.5031568  -0.8803194   0.05012048 ... -0.49182656 -0.3222476
   -0.5626902 ]]]

"""

