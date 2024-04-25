config={
  "type_vocab_size": 0, 
  'use_bias':0,
  'o_bias':0,
}

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
import keras_nlp
from bert4keras3.models import build_transformer_model
from bert4keras3.snippets import sequence_padding
keras.config.set_dtype_policy("bfloat16")
model_name = "Meta-Llama-3-8B-Instruct"
import torch
try:
    os.makedirs(model_name)
except:
    pass
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig

model = AutoModelForCausalLM.from_pretrained('NousResearch/'+model_name,
                                             device_map="cpu",
                                             torch_dtype=torch.bfloat16, 
                                             _attn_implementation = 'eager',
                                             trust_remote_code=False).eval()

tokenizer = AutoTokenizer.from_pretrained('NousResearch/'+model_name)
qw2_config = AutoConfig.from_pretrained('NousResearch/'+model_name)

config[ "vocab_size"]= qw2_config.vocab_size
config[ "num_hidden_layers"]=qw2_config.num_hidden_layers
config[ "query_head"]=qw2_config.num_attention_heads
config[ "num_attention_heads"]=qw2_config.num_key_value_heads
config[ "hidden_size"]=qw2_config.hidden_size
config[ "intermediate_size"]=qw2_config.intermediate_size
config[ "attention_head_size"]=qw2_config.hidden_size//qw2_config.num_attention_heads
config[ "dropout_rate"]=qw2_config.attention_dropout
config[ "hidden_act"]=qw2_config.hidden_act
config[ "max_wavelength"]=qw2_config.rope_theta
import json
with open(model_name+'/config.json', 'w') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)

self = build_transformer_model(
        config_path=model_name+'/config.json',
        model='llama',
        return_keras_model=False,
        with_lm='linear',
    )
QW= self.model
QW.eval()

import numpy as np
input_ids,mask = tokenizer('计算量决定了网络执行时间的长短，参数量决定了占用显存的量').values()
input_ids = keras.ops.expand_dims(input_ids,0)
mask = keras.ops.expand_dims(mask,0)

QW.summary()
print(print(sum(p.numel() for p in model.parameters())))

weights=model.state_dict()
def get_weight(key):
    weight=weights.pop(key)
    return weight

embeddingweight = get_weight('model.embed_tokens.weight')
enocder_embeding = QW.get_layer('Embedding-Token')
enocder_embeding.set_weights([embeddingweight])
for index in range(qw2_config.num_hidden_layers):
    attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
    feed_forward_name = 'Transformer-%d-FeedForward' % index
    att_ln =QW.get_layer('%s-Norm' % attention_name)
    att = QW.get_layer(attention_name)
    ffn_ln = QW.get_layer('%s-Norm' % feed_forward_name)
    ffn = QW.get_layer(feed_forward_name)

    att_ln.set_weights([get_weight('model.layers.'+str(index)+'.input_layernorm.weight')])
    ffn_ln.set_weights([get_weight('model.layers.'+str(index)+'.post_attention_layernorm.weight')])

    o1 = get_weight('model.layers.'+str(index)+'.mlp.gate_proj.weight').T
    o2 = get_weight('model.layers.'+str(index)+'.mlp.up_proj.weight').T
    o3 = get_weight('model.layers.'+str(index)+'.mlp.down_proj.weight').T

    ffn.set_weights([o1,o2,o3])

    q = get_weight('model.layers.'+str(index)+'.self_attn.q_proj.weight').T
    
    k = get_weight('model.layers.'+str(index)+'.self_attn.k_proj.weight').T
    
    v = get_weight('model.layers.'+str(index)+'.self_attn.v_proj.weight').T
    
    o = get_weight('model.layers.'+str(index)+'.self_attn.o_proj.weight').T

    att.set_weights([q,
                     k,
                     v,
                     o])
out_norm = get_weight('model.norm.weight')
QW.get_layer('Output-Norm').set_weights([out_norm])

lm_weights = get_weight('lm_head.weight').T
QW.get_layer('Decoder-Output-LM').set_weights([lm_weights])
QW.save_weights(model_name+'/model.weights.h5')
print('saving')
x1 = model.forward(input_ids.cpu(),attention_mask=mask.cpu())
x2 = QW(input_ids)


print(keras.ops.mean(keras.ops.abs(x2.cpu()-x1.logits.cpu()),-1))
