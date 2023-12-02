# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:09:53 2023

@author: Administrator
"""

#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
from bert4keras3.backend import keras,ops
from bert4keras3.models import build_transformer_model
from bert4keras3.tokenizers import Tokenizer
from bert4keras3.snippets import to_array
#bert from 
#model download form https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-4_H-312_A-12.zip
config_path = 'models/chinese_roberta_L-4_H-312_A-12/bert_config.json'
checkpoint_path = 'models/chinese_roberta_L-4_H-312_A-12/bert_model.ckpt'
dict_path = 'models/chinese_roberta_L-4_H-312_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
token_ids, segment_ids = to_array([token_ids], [segment_ids])

print('\n ===== predicting =====\n')
print(model.predict([token_ids, segment_ids]))
model.predict([np.zeros([1024,32]),np.zeros([1024,32])],batch_size=8)
model.predict([np.zeros([1024,32]),np.zeros([1024,32])],batch_size=8,verbose=1)


"""
keras2.3.1 with tf2.2 ：
gpu:
1024/1024 [==============================] - 0s 466us/step
cpu:
1024/1024 [==============================] - 3s 3ms/step
array([[[-0.529619  , -0.08855614,  0.37196752, ..., -0.33411935,
          0.13711435, -0.956624  ],
        [-0.83184814,  0.3324544 , -0.49911997, ..., -0.12592706,
         -0.87739134, -1.120965  ],
        [-0.9416604 ,  0.11662968,  0.92229784, ...,  0.6774571 ,
          1.5154107 , -0.16043526],
        [-0.891538  , -0.8726713 , -1.5886593 , ...,  0.2074936 ,
         -0.44794142, -1.0378699 ],
        [-0.87546647,  0.75775445, -0.2165907 , ...,  0.63286835,
          2.0644133 , -0.0790057 ],
        [-0.26717812, -0.5348375 ,  0.16076468, ..., -0.9300951 ,
          1.2696625 , -1.60602   ]]], dtype=float32)

keras3 torch backend
gpu:128/128 ━━━━━━━━━━━━━━━━━━━━ 3s 26ms/step  
cpu:128/128 ━━━━━━━━━━━━━━━━━━━━ 6s 45ms/step
[[[-0.52961934 -0.08855701  0.37196732 ... -0.33411875  0.13711427
   -0.9566241 ]
  [-0.8318479   0.33245584 -0.49911973 ... -0.12592846 -0.8773916
   -1.1209643 ]
  [-0.9416604   0.11662959  0.9222967  ...  0.67745614  1.515408
   -0.16043454]
  [-0.8915373  -0.87267166 -1.5886595  ...  0.20749295 -0.44793993
   -1.037869  ]
  [-0.8754667   0.7577548  -0.21659094 ...  0.6328681   2.064412
   -0.07900612]
  [-0.26717797 -0.5348387   0.16076465 ... -0.93009293  1.2696612
   -1.6060183 ]]]

keras3 jax backend
cpu 128/128 ━━━━━━━━━━━━━━━━━━━━ 4s 28ms/step
[[[-0.5296185  -0.08855577  0.37196666 ... -0.3341192   0.13711472
   -0.9566229 ]
  [-0.8318465   0.33245653 -0.49912187 ... -0.12592922 -0.87738854
   -1.1209626 ]
  [-0.9416592   0.11662999  0.92229736 ...  0.67745733  1.5154091
   -0.16043147]
  [-0.89153856 -0.8726713  -1.5886576  ...  0.20749114 -0.4479399
   -1.0378699 ]
  [-0.8754649   0.7577545  -0.21659109 ...  0.6328678   2.0644124
   -0.07900498]
  [-0.26717943 -0.53483707  0.1607644  ... -0.93009156  1.2696632
   -1.6060191 ]]]
conda create -n keras-tf python=3.10
conda activate keras-tf
pip install -r requirements-tensorflow-cuda.txt
pip install keras==3.0

keras3 tf backend
cpu 128/128 ━━━━━━━━━━━━━━━━━━━━ 3s 20ms/step
[[[-0.52961934 -0.08855658  0.37196696 ... -0.3341183   0.13711505
   -0.956624  ]
  [-0.8318459   0.3324545  -0.49912187 ... -0.1259276  -0.8773911
   -1.1209633 ]
  [-0.941661    0.11663001  0.9222968  ...  0.67745817  1.5154085
   -0.1604334 ]
  [-0.89153755 -0.8726738  -1.5886576  ...  0.20749217 -0.4479404
   -1.0378699 ]
  [-0.87546563  0.7577539  -0.21659082 ...  0.63286823  2.0644112
   -0.07900462]
  [-0.26717886 -0.53484     0.16076434 ... -0.93009007  1.2696624
   -1.6060195 ]]]

"""

