# bert4keras3
## 背景
bert4keras是我最喜欢的库之一，但在现在来说其后端tf1.15显得有点落后，因此本库目的在实现对其的升级


## 目的
兼容tf.keras keras2.3.1+tf1.15-tf2.2 以及keras3及其对应后端
目前已经成功实现了bert4keras所支持的所有预训练模型的兼容  
bert4keras实现的优化器目前暂时不做兼容，除开优化器部分外，如何使用请参考bert4keras的example，本仓库的example只提供了如何把模型load出来的测试   

## 权重

目前支持加载的权重：
- <strong>Google原版bert</strong>: https://github.com/google-research/bert
- <strong>brightmart版roberta</strong>: https://github.com/brightmart/roberta_zh
- <strong>哈工大版roberta</strong>: https://github.com/ymcui/Chinese-BERT-wwm
- <strong>Google原版albert</strong><sup><a href="https://github.com/bojone/bert4keras/issues/29#issuecomment-552188981">[例子]</a></sup>: https://github.com/google-research/ALBERT
- <strong>brightmart版albert</strong>: https://github.com/brightmart/albert_zh
- <strong>转换后的albert</strong>: https://github.com/bojone/albert_zh
- <strong>华为的NEZHA</strong>: https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow
- <strong>华为的NEZHA-GEN</strong>: https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow
- <strong>自研语言模型</strong>: https://github.com/ZhuiyiTechnology/pretrained-models
- <strong>T5模型</strong>: https://github.com/google-research/text-to-text-transfer-transformer
- <strong>GPT_OpenAI</strong>: https://github.com/bojone/CDial-GPT-tf
- <strong>GPT2_ML</strong>: https://github.com/imcaspar/gpt2-ml
- <strong>Google原版ELECTRA</strong>: https://github.com/google-research/electra
- <strong>哈工大版ELECTRA</strong>: https://github.com/ymcui/Chinese-ELECTRA
- <strong>CLUE版ELECTRA</strong>: https://github.com/CLUEbenchmark/ELECTRA
- <strong>LaBSE（多国语言BERT）</strong>: https://github.com/bojone/labse
- <strong>Chinese-GEN项目下的模型</strong>: https://github.com/bojone/chinese-gen
- <strong>T5.1.1</strong>: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md#t511
- <strong>Multilingual T5</strong>: https://github.com/google-research/multilingual-t5/
- <strong>GAU-ALPHA</strong>: https://github.com/ZhuiyiTechnology/GAU-alpha

<strong>注意事项</strong>
- 注1：brightmart版albert的开源时间早于Google版albert，这导致早期brightmart版albert的权重与Google版的不完全一致，换言之两者不能直接相互替换。为了减少代码冗余，bert4keras的0.2.4及后续版本均只支持加载<u>Google版</u>以brightmart版中<u>带Google字眼</u>的权重。如果要加载早期版本的权重，请用<a href="https://github.com/bojone/bert4keras/releases/tag/v0.2.3">0.2.3版本</a>，或者考虑作者转换过的<a href="https://github.com/bojone/albert_zh">albert_zh</a>。
- 注2：下载下来的ELECTRA权重，如果没有json配置文件的话，参考<a href="https://github.com/ymcui/Chinese-ELECTRA/issues/3">这里</a>自己改一个（需要加上`type_vocab_size`字段）。
## 2023.12.30发布bert4keras3的第一个版本 
对bert4keras除优化器部分外的升级，实现对tf1.15-tf2.14，jax，torch的多后端兼容


