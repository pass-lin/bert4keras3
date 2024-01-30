# -*- coding: utf-8 -*-
"""
此代码在torch-gpu和jax-cpu中测试通过

"""
import os
os.environ["KERAS_BACKEND"] = "torch"

#jax的话第一次跑是很慢的（因为要编译），但是第二次跑在128长度下相较于torch提速一倍

search_mode='topk'#topp搜搜索
k=100#如果是topk,k应该是>1的整数，如果是topp，k∈(0,1]
end_token=1#chatyuan用的是1 作为结束token
max_len=128#生成最大长度
progress_print=True#是否打印生成的进度，jax好像无效
index_bias=1#如果和t5一样开头是0，那就要选择为1，否则是0
input_lengths=[None,max_len]#代表输入的最大长度，第一个的None是说encoder的输入不限长度

import torch
from bert4keras3.tokenizers import SpTokenizer
from bert4keras3.snippets import sequence_padding
from bert4keras3.backend import keras
from bert4keras3.models import *
np.random.seed(45)
tokenizer = SpTokenizer('ChatYuan-keras/spiece.model', token_start=None, token_end='</s>')
#ckpt是为了兼容苏神的模型，对于新模型我们采用keras3的weights.h5格式存储权重
t5 =build_transformer_model(
    config_path='ChatYuan-keras/config.json',
    model='mt5.1.1',
    return_keras_model=False,
    with_lm='softmax',
    name='T5',
)
#build的时候不需要添加checkpoints_path,在load后用此方法加载
t5.model.load_weights('ChatYuan-keras/T5weights.weights.h5')
#构建cache模型
cache_model=t5.build_cache_model(input_lengths,end_token=1,
                       search_mode=search_mode,k=k,progress_print=True,index_bias=1)

#从bert4torch抄过来的example
e_in=["帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准。",
        "你能干什么？",
        "一个关于冬天的诗歌:"]
for i in range(len(e_in)):
    e_in[i]=tokenizer.encode(e_in[i])[0]
e_in = sequence_padding(e_in)
d_in = np.repeat([[0,3]+[0]*(max_len-2)],len(e_in),0)#decoder的输入要padding到最大长度
#预测
outs = cache_model.predict([e_in,d_in])
#输出
for out in outs:
    print(tokenizer.decode([int(t) for t in out]))