# bert4keras3
## 背景
bert4keras是我最喜欢的库之一，但在现在来说其后端tf1.15显得有点落后，因此本库目的在实现对其的升级



## 目的
兼容keras3及其对应后端 目前已经成功实现了bert4keras所支持的所有预训练模型的兼容  
bert4keras实现的优化器目前暂时不做兼容，除开优化器部分外，如何使用请参考bert4keras的example，本仓库的example只提供了如何把模型load出来的测试  

## 安装
pip install bert4keras3


## 后端版本

keras3的相关后端，请参考keras官方的版本要求  
即使你不使用tf作为后端，也要安装一个tf的cpu版本    
https://github.com/keras-team/keras/blob/master/requirements-jax-cuda.txt  
https://github.com/keras-team/keras/blob/master/requirements-tensorflow-cuda.txt    
https://github.com/keras-team/keras/blob/master/requirements-torch-cuda.txt  

## 功能
初始版本与bert4keras基本相同，可以参考https://github.com/bojone/bert4keras  
但需要注意的是，如果bert4keras的example中必须要要tf.keras的，在本库中依然需要  
如果你需要使用tf之外的其他后端，需要修改bert4keras中的tf api  
由于优化器部分维护工作量过大，本库放弃了对器优化器的维护。并且以后如果推出优化器功能，只keras3版本  
目前keras3支持原生梯度累积、ema，AdamW等，如果需要什么keras不支持的功能欢迎提issue  
除此之外重计算/gradient_checkpoint功能目前依然不支持keras3  

## 如何实现多版本兼容

如果你只是想兼容torch、tf和jax，那么我建议你使用纯keras的api实现o，参考keras.i。对于精细的算子可以使用keras.ops,如果keras实在没有算子，那你只能提供一个api的三后端实现了  
如果你想兼容keras2和tf.api，因为在keras3中增加了ops系列并且删除了绝大部分keras.backend中的算子操作。因此如果你需要兼容tf2是有一定困难的。  
为了解决这个问题，bert4keras3.ops手动对齐了keras3中的ops，api。所以如果你想要兼容keras2和tf.keras，那么在编写代码时请from bert4keras3 import ops，在keras2中使用的是我们对齐的api，而在keras3中使用的是keras.ops。通过这种方法，你可以很容易地实现更好的兼容性  

## 权重

### 兼容bert4keras支持加载的权重，你可以在本来bert4keras支持的tf.keras、tf1.15-tf2.15和keras3加载：  





  | 模型分类| 模型名称 |  权重链接|支持kv-cache 生成|
  | -------------- | -------------------------- | ------------------------------------------------------------ |-- |
| bert/roberta| Google原版bert|[github](https://github.com/google-research/bert)|√|
| |brightmart版roberta |[github](https://github.com/brightmart/roberta_zh)|√|
| | 哈工大版roberta| [github](https://github.com/ymcui/Chinese-BERT-wwm)|√|
| | 追一开源bert|[github](https://github.com/ZhuiyiTechnology/pretrained-models) |√|
| |LaBSE（多国语言BERT） | [github](https://github.com/bojone/labse)|√|
|albert |谷歌albert |[github](https://github.com/google-research/ALBERT) |x|
| | brightmart版albert| [github](https://github.com/brightmart/albert_zh)|x|
| |苏神转换后的albert |[github](https://github.com/bojone/albert_zh) |x|
| NEZHA|双向NEZHA |[github](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow) |x|
| | 单向NEZHA| [github](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow)|x|
|T5 |谷歌T5 | [github](https://github.com/google-research/text-to-text-transfer-transformer)|√|
| | MT5| [github](https://github.com/google-research/text-to-text-transfer-transformer) |√|
| | 苏神T5-pegasus| [github](https://github.com/ZhuiyiTechnology/t5-pegasus)|√|
| | T5.1.1|[github](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md#t511) |√|
|ELECTRA |Google原版ELECTRA |[github](https://github.com/google-research/electra) |x|
| | 哈工大版ELECTRA| [github](https://github.com/ymcui/Chinese-ELECTRA) |x|
| | CLUE版ELECTRA| [github](https://github.com/CLUEbenchmark/ELECTRA)|x|
|GPT-oai | GPT_OpenAI| [github](https://github.com/bojone/CDial-GPT-tf)|x|
| GPT2-ML|  GPT2-ML|  [github](https://github.com/imcaspar/gpt2-ml)|x|
| GAU| GAU-ALPHA|[github](https://github.com/ZhuiyiTechnology/GAU-alpha) |x|
| Roformer| 苏神原版roformer|[github](https://github.com/ZhuiyiTechnology/roformer) |√|
| | roformer-sim|[github](https://github.com/ZhuiyiTechnology/roformer-sim) |√|
|Roformerv2| 苏神原版roformer-v2|[github](https://github.com/ZhuiyiTechnology/roformer-v2) |√|




### bert4keras3的新增加的模型权重，不再使用ckpt存储      
### 通过build_transformer_model( keras_weights_path='xx.weights.h5'）方法读取权重，只能使用keras3加载
  | 模型分类| 模型名称 |  权重链接|支持kv-cache 生成|数据类型|
  | -------------- | -------------------------- | ------------------------------------------------------------ |-- |---- |
| T5.1.1| ChatYuan|  [百度网盘](https://pan.baidu.com/s/1huwRA2sOIYL8e7OFg0QTcg?pwd=6u59) 提取码：6u59  |  √|FP32|
| | Flan-T5-small| [百度网盘](https://pan.baidu.com/s/1SDjh_c0BQnjW3s0DhwHuyw?pwd=eefr) 提取码：eefr|  √|FP32|
| | Flan-T5-base| [百度网盘](https://pan.baidu.com/s/1hpcF6pNqNMfzGqDMK0GNig?pwd=ukep ) 提取码：ukep|  √|FP32|
| | Flan-T5-large| [百度网盘](https://pan.baidu.com/s/1aUtvyBXffdKyLZgSpMZs3A?pwd=1so0 ) 提取码：1so0|  √|FP32|
| | Flan-T5-xl| [百度网盘](https://pan.baidu.com/s/1Lyf2BcIReQER4jpj1LfMZA?pwd=2epu ) 提取码：2epu|  √|FP32|
| | MT5-large| [百度网盘](https://pan.baidu.com/s/1kIPArbQDZ0gsn-B3hzAjyg?pwd=vvoc ) 提取码：vvoc|  √|FP32|
| | UMT5-small| [百度网盘](https://pan.baidu.com/s/1_e8fifYoJ_qxLYIH_Tyhjw?pwd=618x  ) 提取码：618x|  √|FP32|
| | UMT5-base| [百度网盘](https://pan.baidu.com/s/1ivKPkEVwjK9fOlcrzqfT8Q?pwd=phdf ) 提取码：phdf|  √|FP32|
| | UMT5-xl| [百度网盘](https://pan.baidu.com/s/1QnI8X-zbO2qoi_-cV5tGxA?pwd=3a3h) 提取码：3a3h|  √|FP32|


<strong>注意事项</strong>
- 注1：brightmart版albert的开源时间早于Google版albert，这导致早期brightmart版albert的权重与Google版的不完全一致，换言之两者不能直接相互替换。为了减少代码冗余，bert4keras的0.2.4及后续版本均只支持加载<u>Google版</u>以brightmart版中<u>带Google字眼</u>的权重。如果要加载早期版本的权重，请用<a href="https://github.com/bojone/bert4keras/releases/tag/v0.2.3">0.2.3版本</a>，或者考虑作者转换过的<a href="https://github.com/bojone/albert_zh">albert_zh</a>。(苏神注)
- 注2：下载下来的ELECTRA权重，如果没有json配置文件的话，参考<a href="https://github.com/ymcui/Chinese-ELECTRA/issues/3">这里</a>自己改一个（需要加上`type_vocab_size`字段）。(苏神注)
# 版本更新
## 2023.12.30发布bert4keras3的第一个版本1.0 
对bert4keras除优化器部分外的升级，实现对tensorflow，jax，torch的多后端兼容
## 1.31号更新，发布1.1版本  
转换了chatyuan模型权重（基于t5模型）  
更新了支持批量运算的t5-cache推理版本，详细使用参考[t5-cache的使用example](https://github.com/pass-lin/bert4keras3/blob/main/examples/chatyuan-test-example.py)  。里面较为详细地列出了cache模型要如何使用。  
除了T5，还增加了[bert](https://github.com/pass-lin/bert4keras3/blob/main/examples/test_simbert.py)和
[roformer/roformer-v2](https://github.com/pass-lin/bert4keras3/blob/main/examples/test_simroformer.py)的cache支持，用法和t5一样，example里只是测试一下与greedy是否一致


## 3.17号更新，发布1.2版本  
增加了对weights.h5的读取支持  
增加了lora支持，可以通过设置os.environ["ENABLE_LORA"]='1' 启动lora训练，注意的是除了lora之外的参数全部会被冻结  
增加了flash-attention支持，可以通过设置os.environ["FLASH_ATTN"]='1'使用flash-attention  
但是需要注意的是，tensorflow不支持。而jax在https://github.com/nshepperd/flash_attn_jax/releases 下载，torch则是 https://github.com/Dao-AILab/flash-attention  
