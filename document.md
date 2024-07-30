
#目录
- [api说明](#api说明)
  - [bert4keras3.tokenizers](#bert4keras3tokenizers)
    - [Tokenizer类\[同时支持bert4keras\]](#tokenizer类同时支持bert4keras)
      - [Tokenizer.encode方法](#tokenizerencode方法)
      - [Tokenizer.decode方法](#tokenizerdecode方法)
    - [SpTokenizer类\[同时支持bert4keras\]](#sptokenizer类同时支持bert4keras)
  - [bert4keras3.ops](#bert4keras3ops)
  - [bert4keras3.snippets](#bert4keras3snippets)
    - [bert4keras3.sequence\_padding函数\[同时支持bert4keras\]](#bert4keras3sequence_padding函数同时支持bert4keras)
    - [bert4keras3.DataGenerator函数\[同时支持bert4keras\]](#bert4keras3datagenerator函数同时支持bert4keras)
- [bert4keras3.layers](#bert4keras3layers)
    - [bert4keras3.layers.LayerNorms.ScaleOffset\[同时支持bert4keras\]](#bert4keras3layerslayernormsscaleoffset同时支持bert4keras)
      - [bert4keras3.layers.LayerNorms.LayerNormalization\[同时支持bert4keras\]](#bert4keras3layerslayernormslayernormalization同时支持bert4keras)
      - [bert4keras3.layers.LayerNorms.RMSNormalization](#bert4keras3layerslayernormsrmsnormalization)
      - [bert4keras3.layers.LayerNorms.LlamaLayerNorm](#bert4keras3layerslayernormsllamalayernorm)
      - [bert4keras3.layers.LayerNorms.GroupNorm](#bert4keras3layerslayernormsgroupnorm)
    - [bert4keras3.layers.FFN](#bert4keras3layersffn)
      - [bert4keras3.layers.FFN.FeedForward\[同时支持bert4keras\]](#bert4keras3layersffnfeedforward同时支持bert4keras)
      - [bert4keras3.layers.FFN.GemmaFeedForward](#bert4keras3layersffngemmafeedforward)
      - [bert4keras3.layers.FFN.LLamaFeedForward](#bert4keras3layersffnllamafeedforward)
    - [bert4keras3.layers.Embeddings](#bert4keras3layersembeddings)
      - [bert4keras3.layers.Embeddings.PositionEmbedding\[同时支持bert4keras\]](#bert4keras3layersembeddingspositionembedding同时支持bert4keras)
      - [bert4keras3.layers.Embeddings.SinusoidalPositionEmbedding\[同时支持bert4keras\]](#bert4keras3layersembeddingssinusoidalpositionembedding同时支持bert4keras)
      - [bert4keras3.layers.Embeddings.RelativePositionEmbedding\[同时支持bert4keras\]](#bert4keras3layersembeddingsrelativepositionembedding同时支持bert4keras)
      - [bert4keras3.layers.Embeddings.RelativePositionEmbeddingT5\[同时支持bert4keras\]](#bert4keras3layersembeddingsrelativepositionembeddingt5同时支持bert4keras)
      - [bert4keras3.layers.Embeddings.RotaryEmbedding](#bert4keras3layersembeddingsrotaryembedding)
      - [bert4keras3.layers.Embeddings.Embedding](#bert4keras3layersembeddingsembedding)
    - [bert4keras3.layers.GP](#bert4keras3layersgp)
      - [bert4keras3.layers.GP.GlobalPointer](#bert4keras3layersgpglobalpointer)
    - [bert4keras3.layers.Attentions](#bert4keras3layersattentions)
      - [bert4keras3.layers.Attentions.MultiHeadAttention\[同时支持bert4keras\]](#bert4keras3layersattentionsmultiheadattention同时支持bert4keras)
      - [bert4keras3.layers.Attentions.GatedAttentionUnit\[同时支持bert4keras\]](#bert4keras3layersattentionsgatedattentionunit同时支持bert4keras)
    - [bert4keras3.layers.Rwkv\_layer](#bert4keras3layersrwkv_layer)
      - [bert4keras3.layers.Rwkv\_layer.DecomposerDense](#bert4keras3layersrwkv_layerdecomposerdense)
      - [bert4keras3.layers.Rwkv\_layer.TimeShift](#bert4keras3layersrwkv_layertimeshift)
      - [bert4keras3.layers.Rwkv\_layer.TimeShift](#bert4keras3layersrwkv_layertimeshift-1)
      - [bert4keras3.layers.Rwkv\_layer.TimeMix](#bert4keras3layersrwkv_layertimemix)
  - [bert4keras3.backend](#bert4keras3backend)
    - [启动Lora](#启动lora)
    - [启动Flash-attention](#启动flash-attention)
    - [启动tf.keras\[同时支持bert4keras\]](#启动tfkeras同时支持bert4keras)
    - [bert4keras3.backend.flatten\[同时支持bert4keras\]](#bert4keras3backendflatten同时支持bert4keras)
    - [bert4keras3.backend.sequence\_masking\[同时支持bert4keras\]](#bert4keras3backendsequence_masking同时支持bert4keras)
    - [bert4keras3.backend.divisible\_temporal\_padding\[同时支持bert4keras\]](#bert4keras3backenddivisible_temporal_padding同时支持bert4keras)
    - [bert4keras3.backend.root\_mean\_square\[同时支持bert4keras\]](#bert4keras3backendroot_mean_square同时支持bert4keras)
    - [bert4keras3.backend.sinusoidal\_embeddings\[同时支持bert4keras\]](#bert4keras3backendsinusoidal_embeddings同时支持bert4keras)
    - [bert4keras3.backend.align\[同时支持bert4keras\]](#bert4keras3backendalign同时支持bert4keras)
    - [bert4keras3.backend.apply\_rotary\_position\_embeddings\[同时支持bert4keras\]](#bert4keras3backendapply_rotary_position_embeddings同时支持bert4keras)
    - [bert4keras3.backend.multilabel\_categorical\_crossentropy\[同时支持bert4keras\]](#bert4keras3backendmultilabel_categorical_crossentropy同时支持bert4keras)
    - [bert4keras3.backend.multilabel\_categorical\_crossentropy\[同时支持bert4keras\]](#bert4keras3backendmultilabel_categorical_crossentropy同时支持bert4keras-1)
  - [bert4keras3.models](#bert4keras3models)
    - [bert4keras3.models.extend\_with\_language\_model\[同时支持bert4keras\]](#bert4keras3modelsextend_with_language_model同时支持bert4keras)
    - [bert4keras3.models.extend\_with\_unified\_language\_model\[同时支持bert4keras\]](#bert4keras3modelsextend_with_unified_language_model同时支持bert4keras)
    - [bert4keras3.models.build_transformer_model\[同时支持bert4keras\]](#bert4keras3modelstransformer同时支持bert4keras)
    - [bert4keras3.models.Transformer\[同时支持bert4keras\]](#bert4keras3modelstransformer同时支持bert4keras-1)
    - [bert4keras3.models.BERT\[同时支持bert4keras\]](#bert4keras3modelsbert同时支持bert4keras)
    - [bert4keras3.models.NEZHA\[同时支持bert4keras\]](#bert4keras3modelsnezha同时支持bert4keras)
    - [bert4keras3.models.ELECTRA\[同时支持bert4keras\]](#bert4keras3modelselectra同时支持bert4keras)
    - [bert4keras3.models.ALBERT\[同时支持bert4keras\]](#bert4keras3modelsalbert同时支持bert4keras)
    - [bert4keras3.models.ALBERT\_Unshared\[同时支持bert4keras\]](#bert4keras3modelsalbert_unshared同时支持bert4keras)
    - [bert4keras3.models.GPT\[同时支持bert4keras\]](#bert4keras3modelsgpt同时支持bert4keras)
    - [bert4keras3.models.GPT2\[同时支持bert4keras\]](#bert4keras3modelsgpt2同时支持bert4keras)
    - [bert4keras3.models.GPT2\_ML\[同时支持bert4keras\]](#bert4keras3modelsgpt2_ml同时支持bert4keras)
    - [bert4keras3.models.GAU\_alpha\[同时支持bert4keras\]](#bert4keras3modelsgau_alpha同时支持bert4keras)
    - [bert4keras3.models.RoFormer\[同时支持bert4keras\]](#bert4keras3modelsroformer同时支持bert4keras)
    - [bert4keras3.models.RoFormerV2\[同时支持bert4keras\]](#bert4keras3modelsroformerv2同时支持bert4keras)
    - [bert4keras3.models.Gemma](#bert4keras3modelsgemma)
    - [bert4keras3.models.Llama](#bert4keras3modelsllama)
    - [Qwen 模型](#qwen-模型)
    - [bert4keras3.models.T5\_Encoder\[同时支持bert4keras\]](#bert4keras3modelst5_encoder同时支持bert4keras)
    - [bert4keras3.models.T5\_Decoder\[同时支持bert4keras\]](#bert4keras3modelst5_decoder同时支持bert4keras)
    - [bert4keras3.models.T5\[同时支持bert4keras\]](#bert4keras3modelst5同时支持bert4keras)
    - [bert4keras3.models.RWKV6](#bert4keras3modelsrwkv6)


# api说明
为了能同时兼容bert4keras文档的功能，如果bert4keras也支持的内容我会额外标注。但是需要注意的是，所有涉及kv cache的部分bert4keras都是不支持的
## bert4keras3.tokenizers

这一部分主要沿用bert4keras时苏神的实现，对于新的大模型建议使用transformers的AutoTokenizer
### Tokenizer类[同时支持bert4keras]

```python
Tokenizer(
    token_dict,        
    do_lower_case=False,
    word_maxlen=200，
    token_start='[CLS]',
    token_end='[SEP]',
    pre_tokenize=None,
    token_translate=None  
)
```
token_dict:词表的路径  
do_lower_case:是否全部转化为小写 
word_maxlen:token最大长度
token_start：句子开始的特殊符号
token_end：句子结束的特殊符号
pre_tokenize：外部传入的分词函数，用作对文本进行预分词。如果传入pre_tokenize，则先执行pre_tokenize(text)，然后在它的基础上执行原本的tokenize函数。  
例如，在[Wobert](https://github.com/ZhuiyiTechnology/WoBERT)中，苏神使用结巴分词来生成此表，那此时定义tokenzier则为  
```python
import jieba
jieba.initialize()
tokenizer = Tokenizer(
    'vocab.txt',
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)
```
#### Tokenizer.encode方法

```python
Tokenizer.encode(
        first_text,
        second_text=None,
        maxlen=None,
    )

```
first_text：第一句话  
second_text：第二句话  
maxlen：token的最大长度  
return：[token，segment] 其中token是第一句话和第二句话token的拼接。segment是第一句话token一样形状$全是0$的列表和 第二句话token一样形状$全是1$的列表  
#### Tokenizer.decode方法

```python
Tokenizer.decode(
        ids
    )

```
ids：一个一维的列表，由token组成
return:返回token对应的文字

### SpTokenizer类[同时支持bert4keras]
SpTokenizer是T5模型使用的分词器(就是那些.model结尾的文件)  
使用他需要先安装sentencepiece
```
pip3 install sentencepiece
```
使用方法与Tokenizer基本一样,但是有两点需要  
首先T5的token_start和token_end一般为```None```和```</s>```,但具体来说还是要看模型而定的。有的模型也是有token_start的。  
而[Gemma](https://github.com/pass-lin/bert4keras3/blob/main/examples/test-gemma-generate.py)的token_start和token_end为```<bos>```和```<eos>```  
然后，在decode的时候只接受list，并且里面的每个数都是整数  
假如你是np.array或者不是整数可以用下面这种方法解决问题 

```python
ids = np.arrange(10)*1. #假设你有这个一些token
word = tokenizer.decode([int(t) for t in ids])
```

## bert4keras3.ops
当你使用keras3的时候，等价于```from keras import ops```  
具体api用法请参考[keras文档](https://keras.io/api/ops/)   
这里主要是出于兼容[bert4keras](https://github.com/bojone/bert4keras)在tf1.15-tf2.10下使用的。主要是把tf的api和ops的api做了一个对齐。  
如果你想构建一个兼容keras2的模型，那可以考虑使用```from bert4keras3 import ops```代替```from keras import ops```  
但如果你不在乎keras2/tf.keras，可以不用管  
此处主要是为了保证对bert4keras的向后兼容，对于新更新的模型$不做keras2兼容保证$(毕竟谷歌都不兼容了我能咋办)  

## bert4keras3.snippets
这里主要是一些苏神自己定义使的工具方法   
因此在下面我只介绍我自己常用的以及我添加的方法(有的话)  
 ### bert4keras3.sequence_padding函数[同时支持bert4keras]

 ```python
 def sequence_padding(
    inputs, 
    length=None, 
    value=0, 
    seq_dims=1, 
    mode='post',
    show_tqdm=False#bert4keras不支持这个参数
    )
 ```
 批量padding函数输入转化为在axis维形状相同的函数  
 inputs：输入，可以是list也可以是array  
 length：None的时候padding为axis维里最大的长度，不为None的时候padding到length长  
 value：填充值  
 seq_dims：填充的维度  
 mode：输入'pre'是左padding，默认的'post'是右padding
 show_tqdm：要不要显示padding的进度条  
 
  ### bert4keras3.DataGenerator函数[同时支持bert4keras]
 ```python
 DataGenerator( 
    data, 
    batch_size=32, 
 )
 ```
  DataGenerator是一个数据生成器基类，用于在训练深度学习模型时提供批量数据。它能够处理不同类型的数据集，并提供了灵活的批量大小和缓冲区大小的设置  
  data:处理好的数据  
  batch_size：无需多言  
  使用示例,参考[使用albert的文本分类任务](https://github.com/bojone/bert4keras/blob/master/examples/task_sentiment_albert.py).假设下面是使用这个任务
   ```python

def load_data(texts,labels):
    #texts是字符串的列表，labels是标签的列表
    datas = []
    for i,t in enumerate(texts):
        #获取token和segment，使用上面的Tokenizer类
        token,segment = tokenizer.encode(t)
        datas.append([token,segment,labels[i]])
    return datas
datas = load_data(texts,labels)
class data_generator(DataGenerator):
    """数据生成器
    """
    #主要使用方法就是重载这个类
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (token_ids, segment_ids, label) in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                #输出前需要padding成统一的形状
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
#获得数据生成器
generator = data_generator(tdata, batch_size)

#使用数据生成器训练模型
model.fit(
        generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
    )
   ```
必须要注意的是，这个代码不是多线程的。只适合用作微调小规模的数据。如果你的瓶颈在数据读取这块，建议你使用torch的dataloader或者tf.data.具体教程参考[keras的fit方法使用](https://keras.io/api/models/model_training_apis/#fit-method)
# bert4keras3.layers
这里面是我自己定义的keras layers，可同时支持jax tensorflow 和torch。对于tf.keras，keras2和tf1，不提供兼容保证，但理论上是支持的。  
由于本质上是keras的层，所以下面我会用[keras文档](https://keras.io/api/layers/)的风格来介绍每一个layers  
如果没有专门提及，那call方法只接受和输入形状一样的输入

### bert4keras3.layers.LayerNorms.ScaleOffset[同时支持bert4keras]
 ```python
 ScaleOffset( 
    scale=True,
    offset=True,
 )
 ```
 简单的仿射变换层（最后一维乘上gamma向量并加上beta向量）  
说明：1、具体操作为最后一维乘上gamma向量并加上beta向量；  
    2、如果直接指定scale和offset，那么直接常数缩放和平移；  
    3、hidden_*系列参数仅为有条件输入时(conditional=True)使用，  
    用于通过外部条件控制beta和gamma。  

scale: 是否使用缩放向量（gamma）   
offset: 是否使用偏移向量（beta）   

输入形状：(batch_size, …, input_dim)  
输出形状：与输入一样，(batch_size, …, input_dim)
#### bert4keras3.layers.LayerNorms.LayerNormalization[同时支持bert4keras]
 ```python
 LayerNormalization( 
    scale=True,
    offset=True,
    epsilon=None，
 )
 ```
 zero_mean: 是否将输入数据的平均值归一化为0。   
 unit_variance: 是否将输入数据的方差归一化为1。    
 epsilon: 浮点数，用于在计算方差时防止除以零，默认为K.epsilon()。  

 输入输出形状和 ScaleOffset一致  

#### bert4keras3.layers.LayerNorms.RMSNormalization
  ```python
 RMSNormalization( 
    epsilon=None，
 )
 ```
 等价于 ```LayerNormalization(scale=False) ```
#### bert4keras3.layers.LayerNorms.LlamaLayerNorm
  ```python
 LlamaLayerNorm( 
    epsilon=None，
 )
 ```
 与 RMSNormalization基本一致，区别是会先转成fp32再做ln运算再转回来
#### bert4keras3.layers.LayerNorms.GroupNorm
```python
class GroupNorm(Layer):
    def __init__(self,hidden_size,head_size,epsilon=64*1e-5):
```
hidden_size:正整数，输入张量的维度  
head_size：每个group的维度必须能和hidden_size整除  
 ### bert4keras3.layers.FFN
#### bert4keras3.layers.FFN.FeedForward[同时支持bert4keras]
  ```python
FeedForward( 
    units,
    activation='relu',
    use_bias=True,
    kernel_initializer='glorot_uniform',
 )
  ```

units: 正整数，FFN的隐藏维度。  
activation: 字符串或字符串列表，表示第一个Dense层的激活函数。如果是列表，那就常见和列表同等长度且对应激励函数的Dense。这些Dense的输出维度是units，最后通过点乘连接。   
use_bias: Dense层是否使用bias。  
kernel_initializer: Dense层权重矩阵的初始化器。

输入形状： (batch_size, …, input_dim)。

输出形状： 形状与输入相同。

#### bert4keras3.layers.FFN.GemmaFeedForward
  ```python
GemmaFeedForward( 
    units,
    use_bias=True,
    kernel_initializer='glorot_uniform',
 )
  ```
  参数含义，输入输出与FeedForward相同。  
  但是这里写死了记录函数，等价于
  ```python
  FeedForward( 
    units=units,
    activation=['gelu','linear'],
    use_bias=use_bias,
    kernel_initializer=kernel_initializer)
```
但区别是这里的gelu会把approximate设置为True  
#### bert4keras3.layers.FFN.LLamaFeedForward
  ```python
LLamaFeedForward( 
    units,
    activation='relu',
    use_bias=True,
    kernel_initializer='glorot_uniform',
 )
  ```
  参数含义，输入输出与GemmaFeedForward基本一致。
  但是GemmaFeedForward的activation是写死为gelu的，而这里的activation可以自行设置  
  除此之外LLamaFeedForward会先转化为fp32再计算激励函数 

   ### bert4keras3.layers.Embeddings
#### bert4keras3.layers.Embeddings.PositionEmbedding[同时支持bert4keras]
```python
PositionEmbedding( 
    input_dim,
    output_dim,
    merge_mode='add',
    hierarchical=None,
    embeddings_initializer='zeros',
    custom_position_ids=False,
 )
  ```
Bert使用的绝对位置编码  
input_dim: 表示输入序列的最大长度。 
output_dim: 表示位置嵌入的维度。 
merge_mode: 字符串，表示位置嵌入与输入数据的合并方式。可以是 ‘add’（加法）、‘mul’（乘法）、‘zero’（直接返回位置向量）   
hierarchical: 布尔值或浮点数，表示是否使用[分层位置嵌入](https://kexue.fm/archives/7947)。如果是浮点数，表示分层嵌入的混合系数。   
embeddings_initializer: 初始化器，用于位置嵌入矩阵的初始化。   custom_position_ids: 表示是否使用自定义的位置ID。  

输入形状： 如果custom_position_ids为False，输入形状为：(batch_size, seq_len, input_dim)； 如果custom_position_ids为True，输入形状为：[(batch_size, seq_len, input_dim), (batch_size, seq_len)]。

输出形状： (batch_size, seq_len, output_dim)
#### bert4keras3.layers.Embeddings.SinusoidalPositionEmbedding[同时支持bert4keras]
```python
SinusoidalPositionEmbedding( 
    output_dim,
    merge_mode='add',
    custom_position_ids=False,
 )
  ```
[Sinusoidal位置编码](https://kexue.fm/archives/8231)  
output_dim: 整数，表示位置嵌入的维度。   
merge_mode: 字符串，表示位置嵌入与输入数据的合并方式。可以是 ‘add’（加法）、‘mul’（乘法）、‘zero’（直接返回位置向量）    
 custom_position_ids: 布尔值，表示是否使用自定义的位置ID。

输入形状： 如果custom_position_ids为False，形状为：(batch_size, seq_len, input_dim)； 如果custom_position_ids为True，形状为：[(batch_size, seq_len, input_dim), (batch_size, seq_len)]。

输出形状： (batch_size, seq_len, output_dim)
#### bert4keras3.layers.Embeddings.RelativePositionEmbedding[同时支持bert4keras]
```python
RelativePositionEmbedding( 
    input_dim, 
    output_dim, 
    embeddings_initializer='zeros'
 )
 ```
 [第一代相对位置编码](https://arxiv.org/abs/1803.02155)  
input_dim: 整数，表示输入序列的最大长度。  
output_dim: 整数，表示位置嵌入的维度。  
embeddings_initializer: 初始化器，用于位置嵌入矩阵的初始化。

输入形状： 两个张量, ```[(batch_size, seq_len1, dims1),(batch_size, seq_len2, dims2)] ```

输出形状： 和绝对位置编码不同，这里只会直接返回位置编码。```形状是(seq_len1, seq_len2, output_dim)```
#### bert4keras3.layers.Embeddings.RelativePositionEmbeddingT5[同时支持bert4keras]

```python
RelativePositionEmbeddingT5( 
    input_dim,
    output_dim,
    max_distance=128,
    bidirectional=True,
    embeddings_initializer='zeros',
 )
 ```
 [T5的相对位置编码](https://arxiv.org/abs/1910.10683)  
  input_dim: 整数，表示输入序列的最大长度。  
  output_dim: 整数，表示位置嵌入的维度。  
  max_distance: 整数，表示位置差的最大值。  
  bidirectional: 布尔值，表示是否双向计算相对位置。  
  embeddings_initializer: 初始化器，用于位置嵌入矩阵的初始化。

输入形状： 两个张量, ```[(batch_size, seq_len1, dims1),(batch_size, seq_len2, dims2)] ```

输出形状： 和绝对位置编码不同，这里只会直接返回位置编码。```形状是(seq_len1, seq_len2, output_dim)```
#### bert4keras3.layers.Embeddings.RotaryEmbedding
```python
RotaryEmbedding( 
    max_wavelength=10000,
    scaling_factor=1.0,
    sequence_axis=1,
    feature_axis=-1,
 )
 ```
苏神相对位置编码的谷歌实现版本，主要是增加了一些调参的选项  
max_wavelength: 整数。正弦/余弦曲线的最大角波长。
scaling_factor: 浮点数。用于缩放频率范围的缩放因子。
sequence_axis: 整数。输入张量中的序列轴。
feature_axis: 整数。输入张量中的特征轴。

$call方法$的输入  
```call(inputs,start_index=None)```  
inputs: 要应用嵌入的张量输入。这可以具有任何形状，但必须包含序列和特征轴。将对 inputs 应用旋转嵌入并返回。  
start_index: 主要是用于kv-cache，表示现在是第n个token  


输入形状：(batch_size, ..., dim)  
输出形状：同输入  

#### bert4keras3.layers.Embeddings.Embedding

与keras实现的[Embedding](https://keras.io/api/layers/core_layers/embedding/)基本一致，区别是在 call可以传入参数```call(inputs,mode='dense')```。这个参数得以实现transformer的输入和输出embedding共享参数
### bert4keras3.layers.GP
#### bert4keras3.layers.GP.GlobalPointer
```python
GlobalPointer( 
    heads,
    head_size,
    RoPE=True,
    use_bias=True,
    tril_mask=True,
    kernel_initializer='lecun_normal',
 )
  ```
   [全局指针模块](https://kexue.fm/archives/8373)    
   heads: 整数，多头注意力中的头数。  
    head_size: 整数，每个头的大小。   
    RoPE: 布尔值，是否使用旋转位置编码（Rotary Positional Embedding）。  
    use_bias: 布尔值，是否在密集连接中使用偏置向量。  
    tril_mask: 布尔值，是否排除下三角的计算，以避免对重叠实体的重复打分。  
    kernel_initializer: 字符串或初始化器，用于密集连接的权重矩阵的初始化器。  


输入形状:(batch_size, seqlen, input_dim)  
输出形状:(batch_size, heads, seqlen,  seqlen)

在GP的基础上，苏神提出了一个[改进版的高效GP](https://kexue.fm/archives/8877)
```python
EfficientGlobalPointer
```
使用方法与GlobalPointer完全一致

### bert4keras3.layers.Attentions  
#### bert4keras3.layers.Attentions.MultiHeadAttention[同时支持bert4keras]
```python
MultiHeadAttention( 
    heads,
    head_size,
    out_dim=None,
    key_size=None,
    use_bias=True,
    normalization='softmax',
    attention_scale=True,
    attention_dropout=None,
    return_attention_scores=False,
    kernel_initializer='glorot_uniform',
    o_bias=None,#不支持bert4keras
    query_head=None,#不支持bert4keras
    use_EinsumDense = False,#不支持bert4keras
    rope_mode='su',#不支持bert4keras
    max_wavelength=10_000.0,#不支持bert4keras
    scaling_factor=1.0,#不支持bert4keras
    GQA_mode = 'llama'
 )
  ```
heads：头的数量  
head_size：value的头维度  
out_dim：输出的维度  
key_size：query和key的头维度，如果是None就和head_size一样   
use_bias: 是否使用bias  
normalization：对于注意力矩阵的归一化方法，分别是'softmax'（经典方法），'softmax-fp32'（强制使用float32计算softmax），['softmax_plus'](https://kexue.fm/archives/9019),['squared_relu'](https://arxiv.org/abs/2202.10447)  
attention_scale:布尔值，是否对att做scale  
attention_dropout：attention矩阵的dropout比例，0或者None则没有  
return_attention_scores：布尔值，是否返回注意力矩阵  
kernel_initializer：参数初始化方法  
o_bias：最后的输出OutDense是否使用bias，默认和use_bias保持一致  
query_head：用于实现GroupAttention的，默认和heads一样。如果不一样必须大于heads，且能被heads整除  
use_EinsumDense：Dense的实现方法是使用Dense还是EinsumDense，为了兼容Gemma使用  
rope_mode：两种选项，分别是'su'和'keras'。前者代表苏神的原始实现，后面是keras_nlp的实现，以方便调参  
最后两个参数请参考bert4keras3.layers.Embeddings.RotaryEmbedding  
$call方法$```call(self, inputs, mask=None, **kwargs)```  
因为keras的特性要求，在call的输入里要么全是tensor要么全不是。所以这里inputs代表着所有的输入张量的列表，kwargs代表着运算时涉及的参数  
首先对于kwargs涉及的参数主要是下面几个
```python
a_bias = False
p_bias = None
use_cache = None
is_cache_update_index = None
```
a_bias:布尔值，是否使用casual mask    
p_bias：候选为'rotary'，'typical_relative'，'t5_relative'  
use_cache:布尔值，是否使用kv-cache  
is_cache_update_index：整数或张量，表示当前kv-cache所处的token  
输入inputs较为复杂与上面的几个参数有关  
不管怎么样，inputs肯定有的shape是[query,key,value].  
query的形状是(batch_size, seq_len1, dims1),key和value的形状(batch_size, seq_len2, dims2)      
下面要按照顺序给inputs添加其他输入  
1.然后如果a_bias是True的话，要在后面加上对应casual mask ，shape是[seq_len1,seq_len2]  
2.如果p_bias不是None的那要在上一步的基础上添加输入。p_bias是'rotary'，并且rope_mode是'su',那应该加入bert4keras3.layers.Embeddings.SinusoidalPositionEmbedding(mode='zeros')的输出。  
3.如果use_cache是True，且rope_mode是'keras'，在前面的基础上要随便加上一个tensor。虽然这个张量是没有必要的，但为了兼容以前的代码不得已而为之
4.如果use_cache是True，且rope_mode不是'keras'。那要在前面的基础上给inputs加入kv-cache，shape应该是[bacth_size,2,seqlen,heads, head_size].没错你注意到了，如果key_size和head_size不一样的时候是不能使用kv-cache的。k和v的cache会在第二维stack起来  
5.如果p_bias是'typical_relative'，'t5_relative'，输入应该是bert4keras3.layers.Embeddings.RelativePositionEmbedding和bert4keras3.layers.Embeddings.RelativePositionEmbeddingT5的输出。但是如果你用了kv-cache，那注意输入的第一维和query的长度是一致的  

最后要注意的是如果你是看作bert4keras文档，那就去掉上面关于kv cache的部分即可    
输出形状:[batch_size,seq_len,out_dim]
#### bert4keras3.layers.Attentions.GatedAttentionUnit[同时支持bert4keras]  
```python
GatedAttentionUnit(
    units,
    key_size,
    activation='swish',
    use_bias=True,
    normalization='squared_relu',
    self_attention=True,#不支持bert4keras
    attention_scale=True,
    attention_dropout=None,
    kernel_initializer='glorot_uniform',
    factorization=False#不支持bert4keras
)
```
[GAU](https://kexue.fm/archives/8934)  
units: 隐藏的中间维度  
key_size：头的维度  
activation：激励函数  
use_bias: 是否使用bias  
GQA_mode:GQA有两种实现方法，一种是llama的一种是gemma的。推荐使用默认的llama实现  
normalization：对于注意力矩阵的归一化方法，分别是'softmax'（经典方法），'softmax-fp32'（强制使用float32计算softmax），['softmax_plus'](https://kexue.fm/archives/9019),['squared_relu'](https://kexue.fm/archives/9019)  
attention_scale:布尔值，是否对att做scale  
self_attention:如果是True，那query和key会来自同一个矩阵，通过ScaleOffset制造差异。如果是False，那query和key分别来自两个Dense  
attention_scale:布尔值，是否对att做scale  
attention_dropout：attention矩阵的dropout比例，0或者None则没有  
kernel_initializer：参数初始化方法  
factorization：是否做低秩分解。如果是就会从U*(softmax(qk)V)变成U*(softmax(qk)v@VW).这里不考虑激励函数且normalization为softmax。v和qk是一个维度，再通过一个参数矩阵VW转化为和U相同形状的矩阵  
```
call( inputs,a_bias=None, p_bias=None)
```
和MultiHeadAttention一样，输入inputs与参数有关  
1.如果self_attention是True，那inputs是[x],x的shape是[batch_size,seq_len,dims]。否则inputs和MultiHeadAttention一样是[q,k,v]  
2.如果a_bias是True，要在inputs后加入和MultiHeadAttention一样的casual mask  
3.如果p_bias是'rotary'，那要在MultiHeadAttention后加入bert4keras3.layers.Embeddings.SinusoidalPositionEmbedding(mode='zeros')的输出  
### bert4keras3.layers.Rwkv_layer  
在1.4.0版本里bert4keras3加入了对rwkv6的支持，关于rwkv6本身自己的介绍可以参考原作者的文章https://zhuanlan.zhihu.com/p/694593540  
#### bert4keras3.layers.Rwkv_layer.DecomposerDense

```python
class DecomposerDense(Layer):
    def __init__(self,hidden_size,decomposer_size,use_bias=False,name="decomposed_dense"):
        super(DecomposerDense,self).__init__(name=name)
        self.hidden_size = hidden_size
        self.decomposer_size = decomposer_size
        self.use_bias = use_bias
```
rwkv6相较于rwkv5多了一个lora层。值得就是这个DecomposerDense。所以参数含义和lora一致，hidden_size是高维的维度，decomposer_size是低秩的维度，use_bias是低秩转回高秩是否使用bias。  
中间的激励函数写死为tanh了。 
#### bert4keras3.layers.Rwkv_layer.TimeShift
```python
class TimeShift(Layer):
    def __init__(self,name="time_shift"):
        super(TimeShift, self).__init__(name=name)
    def call(self, inputs,cache_x=None):
        x = ops.pad(inputs,[[0,0],[1,0],[0,0]],constant_values=0.)[:,:-1,:]
        if cache_x is not None:
            x = ops.slice_update(x,[0,0,0],cache_x)
        o = x - inputs
        return o
```
RWKV特有的time-shitf层，定义的时候不需要输入参数。  
如果输入的cache_x是空的情况下，就只对inputs本身做time-shitf。如果cache_x不为空，则inputs的第一个token和cache_x相加。  
假设inputs是一个[b,h,d]的tensor的话，cache_x则应该是一个[b,1,d]的tensor。
#### bert4keras3.layers.Rwkv_layer.TimeShift
```python
class ChannelMix(Layer):
    def __init__(self,hidden_size,expand_size,**kwargs):
        super(ChannelMix, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.expand_size = expand_size
    def call(self, inputs,rnn_mode = False):
        ....
        return output#a tensor has same shape like inputs
```
这是rwkv中类似transoformre ffn的层，因此参数的功能也类似  
hidden_size:输入x的维度  
expand_size:中间层的维度  
主要需要重点说明的是call的参数。众所周知rwkv作为rnn模型，在训练和推理的时候会稍有不同。在训练的时候rwkv有着和transformer类似的行为，模型绝大部分是并行的，而推理的时候因为time-shift的存在，即使是ffn也是类似rnn的模式。  
如果rnn_mode是False，那么inputs应该是一个单独的张量。他会独自执行ffn的功能。  
如果是True，那么inputs是一个由两个tensor组成的list。由于这主要是在推理的时候使用，那么此时第一个的list是一个[b,1,d]的tensor，可想而知的是他独自是没法做time-shift的，因为这需要来自上一个token的信息。自然而然，第二个tensor也是一个[b,1,d]的tensor，而他代表着上一个token的信息。  
#### bert4keras3.layers.Rwkv_layer.TimeMix
```python
class TimeMix(Layer):
    def __init__(self,rwkv_kernel,
    hidden_size,decomposer_size,head_size,
    time_decay_size=64,**kwargs):

```
这里是rwkv的核心层，起到了和self-attention类似的作用，可以看作是一个线性attn  
rwkv_kernel：wkv算子需要一个单独的kernel做计算，我们这里提供了一个多后端的rwkv kernel实现。https://github.com/infiy-quine/RWKV6_Keras_Operator  
hidden_size,decomposer_size这两个参数参考DecomposerDense的说明。   
head_size：rwkv6会把模型分为多个头做计算，和MHA类似，但区别是这里的头只包含token自身的信息。  
time_decay_size：rwkv的w层也会过一个DecomposerDense层，但是他的decomposer_size是由这个参数所定义的。
## bert4keras3.backend

### 启动Lora
```python
os.environ['ENABLE_LORA']='1'
```
加入这个环境变量会为启动模型启动lora训练。即除了lora权重和layerNorm外的参数都停止训练。其中lora主要针对的是Dense和Embeding,Dense包括MHA和FFN里使用的Dense  

### 启动Flash-attention
```python
os.environ['FLASH_ATTN']='1'
```
加入这个环境变量会为启动模型启动Flash-attention训练模型。但是你必须要提前安装好相关的库，[torch版本](https://github.com/Dao-AILab/flash-attention)和[jax版本](https://github.com/nshepperd/flash_attn_jax),而tensorflow暂不支持  


### 启动tf.keras[同时支持bert4keras]
```python
os.environ['TF_KERAS']='1'
```
这里主要是针对老版本tf和bert4keras的兼容，设置这个可以让你使用tf.keras作为后端  

### bert4keras3.backend.flatten[同时支持bert4keras]
```python
flatten(tensor, start=None, end=None)
```
将tensor从start到end的维度展平  
### bert4keras3.backend.sequence_masking[同时支持bert4keras]
```python
sequence_masking(
    x, mask=None, value=0, axis=None, bias=None, return_mask=False
)
```
为序列条件mask的函数   
mask: 形如(batch_size, seq_len)的bool矩阵；  
value: mask部分要被替换成的值，可以是'-inf'或'inf'；  
axis: 序列所在轴，默认为1；  
bias: 额外的偏置项，或者附加的mask；  
return_mask: 是否同时返回对齐后的mask。  

### bert4keras3.backend.divisible_temporal_padding[同时支持bert4keras]  
```python
divisible_temporal_padding(x, n)
```
将一维向量序列右padding到长度能被n整除

### bert4keras3.backend.root_mean_square[同时支持bert4keras] 
```python
root_mean_square(x, axis=None, keepdims=False)
```
均方根，相当于模长的变体
### bert4keras3.backend.sinusoidal_embeddings[同时支持bert4keras]
```python
sinusoidal_embeddings(pos, dim, base=10000)
```
计算pos位置的dim维sinusoidal编码

### bert4keras3.backend.align[同时支持bert4keras]
```python
align(tensor, axes, ndim=None)
```
重新对齐tensor（批量版expand_dims）
axes：原来的第i维对齐新tensor的第axes[i]维；
ndim：新tensor的维度。
### bert4keras3.backend.apply_rotary_position_embeddings[同时支持bert4keras]
```python
apply_rotary_position_embeddings(sinusoidal, *tensors)
```
应用RoPE到tensors中
其中，sinusoidal.shape=[b, n, d]，tensors为tensor的列表，而
tensor.shape=[b, n, ..., d]。
### bert4keras3.backend.multilabel_categorical_crossentropy[同时支持bert4keras]
```python
multilabel_categorical_crossentropy(y_true, y_pred)
```
多标签分类的交叉熵
1. y_true和y_pred的shape一致，y_true的元素是0～1
    的数，表示当前类是目标类的概率；
2. 请保证y_pred的值域是全体实数，换言之一般情况下
    y_pred不用加激活函数，尤其是不能加sigmoid或者
    softmax；
3. 预测阶段则输出y_pred大于0的类；
4. 详情请看：https://kexue.fm/archives/7359 和
    https://kexue.fm/archives/9064 。

### bert4keras3.backend.multilabel_categorical_crossentropy[同时支持bert4keras]
```python
msparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False)
```
稀疏版多标签分类的交叉熵
1. y_true.shape=[..., num_positive]，
    y_pred.shape=[..., num_classes]；
2. 请保证y_pred的值域是全体实数，换言之一般情况下
    y_pred不用加激活函数，尤其是不能加sigmoid或者
    softmax；
3. 预测阶段则输出y_pred大于0的类；
4. 详情请看：https://kexue.fm/archives/7359 。


## bert4keras3.models
### bert4keras3.models.extend_with_language_model[同时支持bert4keras]
```python
def extend_with_language_model(BaseModel)
```
添加下三角的Attention Mask（语言模型用）

### bert4keras3.models.extend_with_unified_language_model[同时支持bert4keras]
```python
def extend_with_unified_language_model(BaseModel)
```
添加UniLM的Attention Mask（Seq2Seq模型用）

### bert4keras3.models.build_transformer_model[同时支持bert4keras]
```python
def build_transformer_model(
    config_path=None,
    checkpoint_path=None,
    model='bert',
    application='encoder',
    return_keras_model=True,
    keras_weights_path=None,#不支持bert4keras
    version=None,
    **kwargs
)
```
config_path:字符串，config的路径  
checkpoint_path：字符串，接受ckpt存储格式的文件路径    
model：模型的类型  
application:主要是对bert类模型扩展用，候选为'lm'和'unilm'  
return_keras_model：返回的是keras的模型还是实例化的Transformer类  
keras_weights_path：字符串，接受weights.h5存储格式的文件路径   
version:主要是针对t5，因为t5有't5.1.1'和't5.1.0'
两种情况
kwargs: 这里指的是bert4keras3.models里所有类的参数都可以通过这个进行传递    
### bert4keras3.models.Transformer[同时支持bert4keras]
```python
class Transformer(object):
    """模型基类
    """
    def __init__(
        self,
        vocab_size,  # 词表大小
        hidden_size,  # 编码维度
        num_hidden_layers,  # Transformer总层数
        num_attention_heads,  # Attention的头数
        intermediate_size,  # FeedForward的隐层维度
        hidden_act=None,  # FeedForward隐层的激活函数
        dropout_rate=None,  # Dropout比例
        attention_dropout_rate=None,  # Attention矩阵的Dropout比例
        embedding_size=None,  # 是否指定embedding_size
        attention_head_size=None,  # Attention中V的head_size
        attention_key_size=None,  # Attention中Q,K的head_size
        sequence_length=None,  # 是否固定序列长度
        keep_tokens=None,  # 要保留的词ID列表
        compound_tokens=None,  # 扩展Embedding
        residual_attention_scores=False,  # Attention矩阵加残差
        ignore_invalid_weights=False,  # 允许跳过不存在的权重
        autoresize_weights=False,  # 自动变换形状不匹配的权重
        layers=None,  # 外部传入的Keras层
        prefix=None,  # 层名前缀
        name=None,  # 模型名称
        o_bias=None,
        penalty = 1.0,
        penalty_window = None,
        max_penalty_range = None,
        temperature = 1.0,
        **kwargs
    )
```
大部分参数代码注释比较完善，需要格外说明的：

hierarchical：默认为None，为True时为使用超长编码(利用层次分解，将bert（Transformer）的最长512的序列长度扩充为512*512，会损失一定精度，
但是微调后可以使用很小的代价恢复性能) [苏神博客](https://kexue.fm/archives/7947 )

residual_attention_scores：是否使用残差Attention矩阵。残差Attention矩阵，给每个Attention矩阵加上前上一层的Attention矩阵，
来源RealFormer[论文](https://arxiv.org/abs/2012.11747 ),目前的实现可能还相对粗糙，欠缺通用性。

ignore_invalid_weights：为是否允许跳过名字不匹配的权重。默认为False，为True时，遇到名字不匹配的层名字时， 会输出一个报错信息，但是程序并不会终止，改层的权重会随机初始化。如果采用weights.h5格式的权重此处无用  

o_bias和query_head的作用参考MultiHeadAttention部分介绍  

下面几个参数是bert4keras3-1.4.0加入的新功能。在build_cache_model生成的cache model中才会起效。但是需要注意的是，使用下面这几个参数要保住with_lm='linear',如果是roformer或者bert使用unilm模型，则with_mlm='linear'    
penalty :生成模型的惩罚系数，可以参考[hf的实现](https://github.com/pass-lin/bert4keras3/blob/main/document.md).输入是1则不执行该参数  
penalty_window ：重复惩罚的窗口，假penalty_window=128，那对于一个1024长度的模型来说会分为8个窗口，每个token解码的时候针对当前窗口之前的token和上一个窗口做重复解码惩罚。如果是None，窗口相当于全部token。  
max_penalty_range ：重复惩罚的次数范围，输入是一个二维的list。比如输入是[2,5]，那么会统计窗口内的token出现次数.会对>=2的次数做惩罚,并且最大值为5  
temperature = 1.0：生成模型解码的温度  
```python
#构建kv-cache生成模型方法，bert4keras没有这个方法，涉及kv cache的请无视
def build_cache_model(
    input_lengths:list,
    end_token,
    search_mode='greedy',
    k=1,
    progress_print=False,
    index_bias=0
    )
```
input_lengths：输入inputs的最大长度，是一个整数列表。长度与model.inputs是一致的，每个内容代表该输入的最大长度。  
例如bert的inputs是token和segment两个，那我的可以这么设置
```python
maxlen=512
input_lengths=[maxlen,maxlen]
```
end_token:解码结束的token，碰到这个token会提前结束  
search_mode：解码的方式，支持'greedy'、'topk'、'topp'三种  
k:search_mode是greedy时无效，是topk时应该是一个大于1的整数，是topp时应该是0-1的浮点数.在1.4.0版本，当使用topp的时候输入可以是一个二维list。如果输入是list，那么第一个数代表原来的p值，第二个数topk的k值。会先使用topk选择前k个再使用topp选择k个中概率的前p个。  
progress_print：在每个推理的step内是否展示进度条，只对torch后端有效  
index_bias：主要是针对t5这种模型，会在decoder把0作为第一个token，所以index_bias设置为1.常见的模型可以不考虑这个

### bert4keras3.models.BERT[同时支持bert4keras]
```python
class BERT(Transformer):
    """构建BERT模型
    """
    def __init__(
        self,
        max_position,  # 序列最大长度
        segment_vocab_size=2,  # segment总数目
        with_pool=False,  # 是否包含Pool部分
        with_nsp=False,  # 是否包含NSP部分
        with_mlm=False,  # 是否包含MLM部分
        hierarchical_position=None,  # 是否层次分解位置编码
        custom_position_ids=False,  # 是否自行传入位置id
        shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
    )
```
需要注意的是，robert模型也是用这个模型，但with_nsp是没有用的  
该模型支持kv cacahe生成，但需要在build_cache_model的时候设置application  
### bert4keras3.models.NEZHA[同时支持bert4keras]
参数和BERT一致，但不支持kv-cache生成
### bert4keras3.models.ELECTRA[同时支持bert4keras]
参数和BERT一致，但不支持kv-cache生成
### bert4keras3.models.ALBERT[同时支持bert4keras]
参数和BERT一致，但不支持kv-cache生成
### bert4keras3.models.ALBERT_Unshared[同时支持bert4keras]
解开ALBERT共享约束，当成BERT用  
参数和BERT一致，但不支持kv-cache生成
### bert4keras3.models.GPT[同时支持bert4keras]
[GPT模型](https://github.com/openai/finetune-transformer-lm)，但参数和bert一致
### bert4keras3.models.GPT2[同时支持bert4keras]
[GPT2模型](https://github.com/openai/gpt-2)，但参数和bert一致
### bert4keras3.models.GPT2_ML[同时支持bert4keras]
[GPT2_ML模型](https://github.com/imcaspar/gpt2-ml)，但参数和bert一致  
GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的原因大概是因为它开源的版本参数量达到了GPT2的15亿参数
### bert4keras3.models.GAU_alpha[同时支持bert4keras]
参数和bert一致，但是max_position参数无效
### bert4keras3.models.RoFormer[同时支持bert4keras]
参数和bert一致，但是max_position参数无效，支持kv-cache生成
### bert4keras3.models.RoFormerV2[同时支持bert4keras]
参数和bert一致，但是max_position参数无效，支持kv-cache生成  
在roformer基础上去掉bias，简化Norm，优化初始化等。
### bert4keras3.models.Gemma  
```python
class Gemma(LM_Mask,RoFormer):
    def __init__(self, 
                 with_lm=True,
                 max_wavelength=10_000.0,
                 scaling_factor=1.0,
                 use_EinsumDense = True,
                 use_bias = False,
                 use_dense_bias=False,
                 input_scale =True,
                 share_emebding=True,
                 rope_mode='keras',
    )
```
参数和bert一致，但是max_position参数无效，并且作为LLM理所当然支持kv-cahce  
但有些参数不大一样  
with_lm:是否输出lm-head，如果是False则没有最后的层。如果是True则是softmax，当然你也可以输出linear  
max_wavelength: 整数。正弦/余弦曲线的最大角波长。
scaling_factor: 浮点数。用于缩放频率范围的缩放因子。
use_EinsumDense：Dense的实现方法是使用Dense还是EinsumDense，
use_bias：attention的dense是否使用bias
use_dense_bias：ffn层是否使用bias
input_scale：在gemma模型里，会对通过embeding层的tensor做一个scale。所以默认要开启  
share_emebding：输入的embeding和最后的lm head是否共享参数  
rope_mode：两种选项，分别是'su'和'keras'。前者代表苏神的原始实现，后面是keras_nlp的实现，以方便调参
### bert4keras3.models.Llama  
```python
class Llama(Gemma):
    def __init__(self, input_scale =False,use_EinsumDense=False,
                 share_emebding=False,**kwargs):
        super(Llama, self).__init__(input_scale=input_scale,
                                    use_EinsumDense=use_EinsumDense,
                                    share_emebding=share_emebding,**kwargs)
        .......
```
llama和gemma整体上差不多，但是有些默认参数要做调整  
### Qwen 模型
[Qwen](https://github.com/search?q=qwen&type=repositories)模型实质上就是llama模型，只不过需要设置```use_bias=True;o_bias=False```  
除此之外需要注意的是QWen 0.5B的share_emebding是False  
### bert4keras3.models.T5_Encoder[同时支持bert4keras]
和Bert的参数一致
### bert4keras3.models.T5_Decoder[同时支持bert4keras]
```python
class T5_Decoder(LM_Mask, T5_Base):
    """Google的T5模型（Decoder）
    """
    def __init__(
        with_lm=True, 
        cross_position_bias=True,
        logit_scale=True, 
        decoder_sequence_length=None)
```
with_lm:是否输出lm-head，如果是False则没有最后的层。如果是True则是softmax，当然你也可以输出linear  
cross_position_bias:在cross_attention处是否用t5位置编码  
logit_scale：最后输出是否scale一下，t5原本这里是True，但是chatyuan是False  
decoder_sequence_length：decoer输入的最大长度  
除此之外参数和bert保持一致，且支持kv-cache生成  
### bert4keras3.models.T5[同时支持bert4keras]
把T5_Encoder和T5_Decoder整合成同一个模型里。这里另外支持一个和T5_Decoder独立的build_cache_model方法。区别是T5_Decoder的build_cache_model方法构建的模型需要你输入encoder输出的向量。  
而T5把encoder也整合进了cache模型，你可以把encoder的token也作为输入 
最后就是如果你在 build_transformer_model中设置return_keras_model=False，那你可以分别获取他的encoder和decoder模型。示例代码如下。
```python
t5 = build_transformer_model(
    config_path=config_path,
    model='mt5.1.1',
    return_keras_model=False,
    keras_weights_path=keras_weights_path)

encoder = t5.encoder
decoder = t5.decoder
```
### bert4keras3.models.RWKV6
```python
class RWKV6(Transformer):
    def __init__(
    self,decomposer_size,
    with_lm=True,
    time_decay_size = 64,
    **kwargs)：
```
定义这部分介绍的参数比较简单，绝大部分都和之前的介绍是一致的。decomposer_size和time_decay_size则可以参考time-mix层的介绍。

```python
def enable_state_tunig(self,
        time_shitf_tuning=False
        ):
    for layer in self.layers.values():
        if isinstance(layer,TimeMix) or isinstance(layer,ChannelMix):
            layer.enable_state_tunig(time_shitf_tuning)
        elif not lora_model:
            layer.trainable = False
```
详细介绍参考原作者的文章https://zhuanlan.zhihu.com/p/695005541  
通过这里来启动state-tuning，rwkv原作者的文章里只对了time-mix层的wkv算子做了state-tuning。但其实rwkv接受来自上一时间的信息不只是wkv算子，还有两个time-shift层也可以接受上一时间的信息。都可以看作来自上一时间的state。  
因此我提供了time_shitf_tuning参数，如果设置为true则可以把time-shift也开启state-tuning。  
并且该部分可以和lora混用。  
需要注意的是，state tuning只能在纯py算子的情况下才能训练，推理的话就无所谓了。如果训练的话建议用jax后端，torch后端慢到不可接受。  
```python
def build_cache_model(self, 
            input_lengths: list, 
            end_token, 
            search_mode='greedy', 
            k=1, 
            progress_print=False, 
            index_bias=0,
            input_state=False,
            output_state=False):
```
构建生成模型的方法，大部分参数都和之前transformer使用的一样。主要增加了input_state和output_state两个参数。  
如果output_state为true，那么模型除了输出生成结果，还会输出state。  
同理如果input_state为true，那么在输入的时候你还需要输入模型的state。  
这里的state指的是wkv层的state，两个time-shift层对应的state。
