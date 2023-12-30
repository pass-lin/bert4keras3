# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 19:09:29 2023

@author: Administrator
"""

#测试一下T5-PEGASUS
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import jieba
from bert4keras3.models import build_transformer_model
from bert4keras3.tokenizers import Tokenizer
from bert4keras3.snippets import AutoRegressiveDecoder
base_path='models/chinese_t5_pegasus_base/'
config_path = base_path+'config.json'
checkpoint_path = base_path+ 'model.ckpt'
dict_path = base_path+ 'vocab.txt'
tokenizer= Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


t5 = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='mt5.1.1',
    return_keras_model=False,
    name='T5',
    dropout_rate=0,
)

encoder = t5.encoder
decoder = t5.decoder
t5.model.summary()
class AutoTitle(AutoRegressiveDecoder):
    def generate(self, text, topk=1):
        c_encoded = encoder.predict(np.array([tokenizer.encode(text)[0]]))[0]
        output_ids=[self.start_id]
        while output_ids[-1]!=self.end_id and len(output_ids)<128:
            
            outs= self.last_token(decoder).predict([np.expand_dims(c_encoded,0),np.reshape(output_ids,[1,-1])],verbose=3)  # 基于beam search
            out=np.argmax(outs)
            output_ids.append(out)
            
        return tokenizer.decode(output_ids).replace(' ','')

    
autotitle = AutoTitle(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=128

)
print(autotitle.generate('针对以超立方体网络为蓝本的多处理机系统的可靠性和容错能力的精准度量问题,结合多处理机系统遭受计算机病毒攻击时常常发生结构性故障的特点,研究了n维超立方体网络的结构连通性和子结构连通性评价问题。首先,使 用构造n维超立方体网络的3路结构割的方法得到其3路结构连通度的一个上界;然后,使用构造n维超立方体网络的3路子结构集的等价变换或约简变换的方法,得到其3路结构子连通度的一个下界;最后,利用任意网络的3路结构连通度不小于3路子结构连通度的性质,证实了超立方体网络的3路结构连通度和子结构连通度均为该超立方体网络维数'))

'''
原版bert4keras的输出是
针对以超立方体网络为蓝本的多处理机系统的可靠性和容错能力的精准度量问题, 结合多处理机系统遭受计算机病毒攻击时常常发生结构性故障的特点, 研究了n维超立方体网络的结构连通性和子结构连通性评价问题。

'''
