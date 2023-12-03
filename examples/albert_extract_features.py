# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:09:53 2023

@author: Administrator
"""

#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "torch"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
from bert4keras3.backend import keras
#from bert4keras3.backend import ops
from bert4keras3.models import build_transformer_model
from bert4keras3.tokenizers import Tokenizer
from bert4keras3.snippets import to_array
#bert from 
base_path='models/albert_tiny_zh_google/'
config_path =base_path+ 'albert_config_tiny_g.json'
checkpoint_path = base_path+'albert_model.ckpt'
dict_path = base_path+'vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path,model='albert')  # 建立模型，加载权重

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
1024/1024 [==============================] - 0s 438us/step
cpu:
1024/1024 [==============================] - 4s 3ms/step
[[[ 0.01209984  0.18839289  0.67528975 ...  0.32766294  0.40439922
   -0.9395135 ]
  [-0.72044194  0.3313576  -0.21194784 ... -0.89638084 -0.5744962
   -0.5753268 ]
  [-1.7922564  -0.7256814   0.73471934 ...  0.6579653   0.3831244
   -1.0155993 ]
  [-1.4994439  -0.9389021  -1.590501   ...  0.44343844  0.26294327
   -0.8904741 ]
  [-1.3181653   0.5725314   0.8430841  ...  0.41200566  2.0104585
   -0.69367486]
  [-0.74496406 -0.00863537  0.32828316 ... -0.7465168   1.0278261
   -0.8846758 ]]]

keras3 torch backend
gpu:128/128 ━━━━━━━━━━━━━━━━━━━━ 3s 21ms/step 
cpu:128/128 ━━━━━━━━━━━━━━━━━━━━ 6s 45ms/step
[[[ 0.01209852  0.18839335  0.67528975 ...  0.32766166  0.40439856
   -0.9395136 ]
  [-0.7204431   0.33135742 -0.21194784 ... -0.8963808  -0.5744967
   -0.57532763]
  [-1.7922562  -0.7256814   0.7347178  ...  0.6579639   0.38312304
   -1.0155989 ]
  [-1.4994448  -0.9389011  -1.5905024  ...  0.4434386   0.26294178
   -0.8904746 ]
  [-1.318164    0.5725332   0.84308225 ...  0.4120072   2.0104592
   -0.6936737 ]
  [-0.74496585 -0.00863451  0.3282838  ... -0.7465183   1.0278255
   -0.8846744 ]]]

keras3 jax backend
cpu 
128/128 ━━━━━━━━━━━━━━━━━━━━ 3s 23ms/step
[[[ 0.012098    0.18839322  0.67529124 ...  0.3276623   0.40439853
   -0.9395129 ]
  [-0.7204414   0.33135745 -0.21194819 ... -0.8963812  -0.57449585
   -0.57532734]
  [-1.792257   -0.7256823   0.73471886 ...  0.6579653   0.3831221
   -1.0155988 ]
  [-1.4994441  -0.938902   -1.5905012  ...  0.4434394   0.26294184
   -0.89047426]
  [-1.3181643   0.5725317   0.8430841  ...  0.41200542  2.010457
   -0.69367546]
  [-0.74496675 -0.00863315  0.32828578 ... -0.74651843  1.0278255
   -0.8846752 ]]]


keras3 tf backend
cpu
128/128 ━━━━━━━━━━━━━━━━━━━━ 2s 19ms/step
[[[ 0.0120999   0.18839282  0.67529106 ...  0.32766137  0.40439838
   -0.93951386]
  [-0.7204417   0.33135727 -0.21194816 ... -0.8963813  -0.5744957
   -0.5753264 ]
  [-1.7922573  -0.7256824   0.73471934 ...  0.6579636   0.3831234
   -1.0155979 ]
  [-1.4994441  -0.9389016  -1.5905008  ...  0.44343853  0.2629423
   -0.8904741 ]
  [-1.3181647   0.5725314   0.84308296 ...  0.41200656  2.0104578
   -0.6936742 ]
  [-0.7449653  -0.00863389  0.3282857  ... -0.7465191   1.0278255
   -0.8846757 ]]]
"""

