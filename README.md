# bert4keras3
## 背景
bert4keras是我最喜欢的库之一，但在现在来说其后端tf1.15显得有点落后，因此本库目的在实现对其的升级



## 目的
兼容keras3及其对应后端 目前已经成功实现了bert4keras所支持的所有预训练模型的兼容  
bert4keras实现的优化器目前暂时不做兼容，除开优化器部分外，如何使用请参考bert4keras的example，本仓库的example只提供了如何把模型load出来的测试  


## api文档
请参考[api说明](https://github.com/pass-lin/bert4keras3/blob/main/document.md)  


## 安装

因为我是个人开发，连草台班子都不是，经常会发布修改bug的版本，所以建议安装最新版本
```
pip install --upgrade bert4keras3
```
### 后端安装
如果你用不是tensorflow后端，我建议安装一个tensorflow-cpu==2.10  
当然如果你不需要加载旧版bert4keras对应的权重的话（对应下面模型权重的第一个表），那其实tf的cpu也不用安装。  
```
pip3 install tensorflow-cpu==2.10
pip3 install --upgrade keras
```
如果你用torch后端，直接安装最新的torch就行了。但是我个人建议torch后端只用来调试  
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install keras
```
如果你需要使用tensorflow后端，那我建议你安装tensorflow的2.15  
```
pip3 install tensorflow[and-cuda]==2.15
pip3 install --upgrade keras
```
当然你想安装最新的也可以，但是问题就是加载苏神的权重会有点问题。谷歌的尿性你们懂的  
还有就是cuda版本要大于12.2，你的服务器不一定能同步。可以看[tensorflow的cuda、cudnn版本对应](https://tensorflow.google.cn/install/source#tested_build_configurations)  
如果你想使用jax后端，jax安装建议看[keras官方文档的jax-cuda要求](https://github.com/keras-team/keras/blob/master/requirements-jax-cuda.txt  )    
比如在keras3.3.3的情况下，官方推荐的版本是jax 0.4.23，那安装可以这么写  
```
#cuda12
pip install -U "jax[cuda12]==0.4.23" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#cuda11
pip install -U "jax[cuda118]==0.4.23" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip3 install --upgrade keras
```
jax和tensorflow后端只能在linux使用cuda  

## 功能
初始版本与bert4keras基本相同，可以参考https://github.com/bojone/bert4keras  
但需要注意的是，如果bert4keras的example中必须要要tf.keras的，在本库中依然需要  
如果你需要使用tf之外的其他后端，需要修改bert4keras中的tf api  
由于优化器部分维护工作量过大，本库放弃了对器优化器的维护。并且以后如果推出优化器功能，只keras3版本  
目前keras3支持原生梯度累积、ema，AdamW等，如果需要什么keras不支持的功能欢迎提issue  
除此之外重计算/gradient_checkpoint功能目前依然不支持keras3  

## 如何实现多版本兼容

如果你只是想兼容torch、tf和jax，那么我建议你使用纯keras的api实现，参考keras.io。对于精细的算子可以使用keras.ops,如果keras实在没有算子，那你只能提供一个api的三后端实现了  
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
  | 模型分类| 模型名称 |  权重链接|数据类型|分词器|
  | -------------- | -------------------------- | ------------------------------------------------------------ |---- |--------  |
| [T5.1.1](https://github.com/pass-lin/bert4keras3/blob/main/examples/chatyuan-test-example.py)  | ChatYuan|  [ModelScope](https://www.modelscope.cn/models/q935499957/ChatYuan-T5-Keras/summary) |  FP32|SpTokenizer|
| | Flan-T5-small| [ModelScope](https://www.modelscope.cn/models/q935499957/flan-t5-small-Keras/summary) |  FP32|SpTokenizer|
| | Flan-T5-base| [ModelScope](https://www.modelscope.cn/models/q935499957/flan-t5-base-Keras/summary) |  FP32|SpTokenizer|
| | Flan-T5-large| [ModelScope](https://www.modelscope.cn/models/q935499957/flan-t5-large-Keras/summary) |  FP32|SpTokenizer|
| | Flan-T5-xl| [ModelScope](https://www.modelscope.cn/models/q935499957/flan-t5-xl-Keras/summary) |  FP32|SpTokenizer|
| | MT5-large| [ModelScope](https://www.modelscope.cn/models/q935499957/mt5-large-Keras/summary) |  FP32|SpTokenizer|
| | UMT5-small| [ModelScope](https://www.modelscope.cn/models/q935499957/umt5-small-Keras/summary) |  FP32|SpTokenizer|
| | UMT5-base| [ModelScope](https://www.modelscope.cn/models/q935499957/umt5-base-Keras/summary) |  FP32|SpTokenizer|
| | UMT5-xl| [ModelScope](https://www.modelscope.cn/models/q935499957/umt5-xl-Keras/summary) |  FP32|SpTokenizer|
|[Gemma](https://github.com/pass-lin/bert4keras3/blob/main/examples/test-gemma-generate.py)| Gemma-2b| [ModelScope](https://www.modelscope.cn/models/q935499957/gemma_2b_en-Keras/summary) |  BF16|SpTokenizer|
|| Gemma-2b-Code| [ModelScope](https://www.modelscope.cn/models/q935499957/code_gemma_2b_en-Keras/summary) |  BF16|SpTokenizer|
|| Gemma-2b-it|[ModelScope](https://www.modelscope.cn/models/q935499957/gemma_instruct_2b_en-Keras/summary) |  BF16|SpTokenizer|
|| Gemma1.1-2b-it| [ModelScope](https://www.modelscope.cn/models/q935499957/gemma_1.1_instruct_2b_en-Keras/summary) |  BF16|SpTokenizer|
||Gemma-7b| [ModelScope](https://www.modelscope.cn/models/q935499957/gemma_7b_en-Keras/summary) |  BF16|SpTokenizer|
|| Gemma-7b-Code| [ModelScope](https://www.modelscope.cn/models/q935499957/code_gemma_7b_en-Keras/summary) |  BF16|SpTokenizer|
|| Gemma-7b-it|[ModelScope](https://www.modelscope.cn/models/q935499957/gemma_instruct_7b_en-Keras/summary) |  BF16|SpTokenizer|
|| Gemma1.1-7b-it| [ModelScope](https://www.modelscope.cn/models/q935499957/gemma_1.1_instruct_7b_en-Keras/summary) |  BF16|SpTokenizer|
|| Gemma-7b-it-Code|[ModelScope](https://www.modelscope.cn/models/q935499957/code_gemma_instruct_7b_en-Keras/summary) |  BF16|SpTokenizer|
|[Llama](https://github.com/pass-lin/bert4keras3/blob/main/examples/test-llama-generate.py)  |Yi-6B | [ModelScope](https://www.modelscope.cn/models/q935499957/Yi-6B-Keras/summary) | BF16|AutoTokenizer|
||Yi-6B-it | [ModelScope](https://www.modelscope.cn/models/q935499957/Yi-6B-Chat-Keras/summary) | BF16|AutoTokenizer|
||Yi-9B | [ModelScope](https://www.modelscope.cn/models/q935499957/Yi-9B-Keras/summary) | BF16|AutoTokenizer|
||Yi-1.5-6B | [ModelScope](https://www.modelscope.cn/models/q935499957/Yi-1.5-6B-Keras/summary) | BF16|AutoTokenizer|
||Yi-1.5-9B | [ModelScope](https://www.modelscope.cn/models/q935499957/Yi-1.5-9B-Keras/summary) | BF16|AutoTokenizer|
||Llama3-8B | [ModelScope](https://www.modelscope.cn/models/q935499957/Meta-Llama-3-8B-Keras/summary) | BF16|AutoTokenizer|
||Llama3-8B-it | [ModelScope](https://www.modelscope.cn/models/q935499957/Meta-Llama-3-8B-Instruct-Keras/summary) | BF16|AutoTokenizer|
||Llama3.1-8B | [ModelScope](https://www.modelscope.cn/models/q935499957/Meta-Llama-3.1-8B-Keras/summary) | BF16|AutoTokenizer|
||Llama3.1-8B-it | [ModelScope](https://www.modelscope.cn/models/q935499957/Meta-Llama-3.1-8B-Instruct-Keras/summary) | BF16|AutoTokenizer|
|[千问](https://github.com/pass-lin/bert4keras3/blob/main/examples/test-Qwen-generate.py)  |Qwen-0.5B | [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen1.5-0.5B-Keras/summary) | BF16|AutoTokenizer|
|| Qwen-0.5B-it| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen1.5-0.5B-Chat-Keras/summary) | BF16|AutoTokenizer|
|| Qwen-1.8B| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen1.5-1.8B-Keras/summary) | BF16|AutoTokenizer|
|| Qwen-1.8B-it| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen1.5-1.8B-Chat-Keras/summary) | BF16|AutoTokenizer|
|| Qwen-4B| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen1.5-4B-Keras/summary) | BF16|AutoTokenizer|
|| Qwen-4B-it| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen1.5-4B-Chat-Keras/summary) | BF16|AutoTokenizer|
|| Qwen-7B| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen1.5-7B-Keras/summary) | BF16|AutoTokenizer|
|| Qwen-7B-it| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen1.5-7B-Chat-Keras/summary) | BF16|AutoTokenizer|
|| Qwen-14B| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen1.5-14B-Keras/summary) | BF16|AutoTokenizer|
|[千问2](https://github.com/pass-lin/bert4keras3/blob/main/examples/test-Qwen-generate.py)| Qwen2-0.5B| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen2-0.5B-Keras/summary) | BF16|AutoTokenizer|
|| Qwen2-0.5B-it| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen2-0.5B-Instruct-Keras/summary) | BF16|AutoTokenizer|
|| Qwen2-1.5B| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen2-1.5B-Keras/summary) | BF16|AutoTokenizer|
|| Qwen2-1.5B-it| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen2-1.5B-Instruct-Keras/summary) | BF16|AutoTokenizer|
|| Qwen2-7B| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen2-7B-Keras/summary) | BF16|AutoTokenizer|
|| Qwen2-7B-it| [ModelScope](https://www.modelscope.cn/models/q935499957/Qwen2-7B-Instruct-Keras/summary) | BF16|AutoTokenizer|
|[RWKV6](https://github.com/pass-lin/RWKV6-Keras)| RWKV6-1.6B| [ModelScope](https://www.modelscope.cn/models/q935499957/RWKV6-1.6B-Keras) | BF16|RWKV_TOKENIZER|
|| RWKV6-3B| [ModelScope](https://www.modelscope.cn/models/q935499957/RWKV6-3B-Keras) | BF16|RWKV_TOKENIZER|
|| RWKV6-7B| [ModelScope](https://www.modelscope.cn/models/q935499957/RWKV6-7B-Keras) | BF16|RWKV_TOKENIZER|
|| RWKV6-12B-it| [ModelScope](https://www.modelscope.cn/models/q935499957/RWKV6-12B-it-Keras) | BF16|RWKV_TOKENIZER|
|| RWKV6-14B| [ModelScope](https://www.modelscope.cn/models/q935499957/RWKV6-14B-Keras) | BF16|RWKV_TOKENIZER|
|[Gemma2](https://github.com/pass-lin/bert4keras3/blob/main/examples/test-gemma2-generate.py)  |Gemma2-2b | [ModelScope](https://www.modelscope.cn/models/q935499957/gemma2_2b_en-Keras) | BF16|AutoTokenizer|
||Gemma2-2b-it | [ModelScope](https://www.modelscope.cn/models/q935499957/gemma2_instruct_2b_en-Keras) | BF16|AutoTokenizer|
||Gemma2-9b | [ModelScope](https://www.modelscope.cn/models/q935499957/gemma2_9b_en-Keras) | BF16|AutoTokenizer|
||Gemma2-9b-it | [ModelScope](https://www.modelscope.cn/models/q935499957/gemma2_instruct_9b_en-Keras) | BF16|AutoTokenizer|
||Gemma2-27b | [ModelScope](https://www.modelscope.cn/models/q935499957/gemma2_27b_en-Keras) | BF16|AutoTokenizer|
||Gemma2-27b-it | [ModelScope](https://www.modelscope.cn/models/q935499957/gemma2_instruct_27b_en-Keras) | BF16|AutoTokenizer|
<!-- || | [百度网盘]() |  BF16|AutoTokenizer| -->



<strong>注意事项</strong>
- 注1：brightmart版albert的开源时间早于Google版albert，这导致早期brightmart版albert的权重与Google版的不完全一致，换言之两者不能直接相互替换。为了减少代码冗余，bert4keras的0.2.4及后续版本均只支持加载<u>Google版</u>以brightmart版中<u>带Google字眼</u>的权重。如果要加载早期版本的权重，请用<a href="https://github.com/bojone/bert4keras/releases/tag/v0.2.3">0.2.3版本</a>，或者考虑作者转换过的<a href="https://github.com/bojone/albert_zh">albert_zh</a>。(苏神注)
- 注2：下载下来的ELECTRA权重，如果没有json配置文件的话，参考<a href="https://github.com/ymcui/Chinese-ELECTRA/issues/3">这里</a>自己改一个（需要加上`type_vocab_size`字段）。(苏神注)
- 注3： 模型分类这里会跳转到使用的example
- 注4：SpTokenizer和RWKV_TOKENIZER来自bert4keras3.tokenizers,AutoTokenizer指的是transformers的分词器。用法不同需要注意  
- 注5：因为不能转换全部的权重，所以我提供了转化权重的脚本，有需要自己去转。
- 注6：bert4keras3的新增加的模型权重均支持kv-cache生成
- 注7: it模型指的是instruct模型，也就是我们俗话说的chat模型
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

## 4.25号更新，发布1.3版本  
重新整理了苏神的代码,更新了对 Gemma,Qwen,和llama系列模型（llama3和Yi）的支持，转换了UMT5，FlanT5的权重，并且提供了转换脚本，大家可以自行转换权重

## 7.30更新，发布1.4版本
新版本可以在[build_transformer_model](https://github.com/pass-lin/bert4keras3/blob/main/document.md#bert4keras3modelsbuild_transformer_model%E5%90%8C%E6%97%B6%E6%94%AF%E6%8C%81bert4keras)时添加penalty，penalty_window ,max_penalty_range,temperature四个参数。  
详情可以参考[文档](https://github.com/pass-lin/bert4keras3/blob/main/document.md#bert4keras3modelstransformer%E5%90%8C%E6%97%B6%E6%94%AF%E6%8C%81bert4keras)。  
增加了RWKV6的层及其模型的支持，关于层的详细介绍可以查看文档[RWKV_layer](https://github.com/pass-lin/bert4keras3/blob/main/document.md#bert4keras3layersrwkv_layer).   
对于RWKV6更详细的介绍，我们单独创建了一个[RWKV6仓库](https://github.com/pass-lin/RWKV6-Keras)，在这里你可以看到关于本库对RWKV6的详细介绍
