config={
  "type_vocab_size": 0, 
}

import os
os.environ['KAGGLE_USERNAME'] = 'your name'
os.environ['KAGGLE_KEY'] = 'your key'

batch_size=14
epochs = 15
os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras_nlp
from bert4keras3.models import build_transformer_model
from bert4keras3.snippets import sequence_padding
keras.config.set_dtype_policy("bfloat16")
model_name = "code_gemma_2b_en"

try:
    os.makedirs(model_name)
except:
    pass
gemma = keras_nlp.models.GemmaCausalLM.from_preset(model_name,preprocessor=None)
gemma.eval()
from keras import ops
backbone = gemma.get_layer('gemma_backbone')
gemma_config = backbone.get_config()
config[ "vocab_size"]=gemma_config['vocabulary_size']
config[ "num_hidden_layers"]=gemma_config['num_layers']
config[ "query_head"]=gemma_config['num_query_heads']
config[ "num_attention_heads"]=gemma_config['num_key_value_heads']
config[ "hidden_size"]=gemma_config['hidden_dim']
config[ "intermediate_size"]=gemma_config['intermediate_dim']//2
config[ "attention_head_size"]=gemma_config['head_dim']
config[ "attention_probs_dropout_prob"]=gemma_config['dropout']
config[ "dropout_rate"]=gemma_config['dropout']
hidden_dim=config[ "hidden_size"]
import json
with open(model_name+'/config.json', 'w') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)

self = build_transformer_model(
        config_path=model_name+'/config.json',
        model='gemma',
        return_keras_model=False,
        with_lm='linear',
    )
MyGemma= self.model
MyGemma.eval()
gemma.summary()


layers_dict = {}
for layer in backbone.layers:
    if layer.weights!=[]:
        layers_dict[layer.name]=layer

def get_weights(layer,i):
    return layer.weights[i].value
embeding_weights  = [get_weights(layers_dict['token_embedding'],0)]
MyGemma.get_layer('Embedding-Token').set_weights(embeding_weights)

fln_weights  = [get_weights(layers_dict['final_normalization'],0)]
MyGemma.get_layer('Output-Norm').set_weights(fln_weights)
from tqdm import tqdm
for i in tqdm(range(gemma_config['num_layers'])):
    block = layers_dict['decoder_block_'+str(i)]
    attention_name = 'Transformer-%d-MultiHeadSelfAttention' % i
    feed_forward_name = 'Transformer-%d-FeedForward' % i

    MyGemma.get_layer('%s-Norm' % attention_name).set_weights([get_weights(block,0)])
    ws=[]
    i=0

    for i in range(3):
        ws.append(ops.reshape(block.get_weights()[1+i],MyGemma.get_layer(attention_name).weights[i].shape))
        #ws.append(ops.reshape(ops.transpose(get_weights(block,i+1),[1,0,2]),MyGemma.get_layer(attention_name).weights[i].shape))
    i+=1
    ws.append(ops.reshape(block.get_weights()[1+i],MyGemma.get_layer(attention_name).weights[i].shape))
    MyGemma.get_layer(attention_name).set_weights(ws)
    ws=[]
    for i in range(3):
        ws.append(ops.reshape(get_weights(block,i+6),MyGemma.get_layer(feed_forward_name).weights[i].shape))
    
    MyGemma.get_layer('%s-Norm' % feed_forward_name).set_weights([get_weights(block,5)])
    MyGemma.get_layer(feed_forward_name).set_weights(ws)

import numpy as np 
MyGemma.save_weights(model_name+'/model.weights.h5')
print('saving')
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset("gemma_2b_en")
b= x = keras.ops.convert_to_numpy([tokenizer( "Where are you ?i am come from china. this is a glory conutry")])
prompt = {
    # Token ids for "<bos> Keras is".
    "token_ids": x,
    # Use `"padding_mask"` to indicate values that should not be overridden.
    "padding_mask": np.ones_like(x),
}

print(gemma.predict(prompt))
print(MyGemma.predict(x))
print(keras.ops.mean(gemma.predict(prompt)-MyGemma.predict(x),-1))

