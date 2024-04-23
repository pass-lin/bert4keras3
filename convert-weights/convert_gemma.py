config={
  "type_vocab_size": 0, 
}

import os
os.environ['KAGGLE_USERNAME'] = 'your-name'
os.environ['KAGGLE_KEY'] = 'your-key'

batch_size=14
epochs = 15
os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras_nlp
from bert4keras3.models import build_transformer_model
from bert4keras3.snippets import sequence_padding
keras.config.set_dtype_policy("bfloat16")
model_name = "code_gemma_instruct_7b_en"

try:
    os.makedirs(model_name)
except:
    pass
gemma = keras_nlp.models.GemmaCausalLM.from_preset(model_name,preprocessor=None)
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
gemma.summary()
MyGemma.summary()
layers_dict = {}
for layer in backbone.layers:
    if layer.weights!=[]:
        layers_dict[layer.name]=layer

embeding_weights  = layers_dict['token_embedding'].get_weights()
MyGemma.get_layer('Embedding-Token').set_weights(embeding_weights)

fln_weights  = layers_dict['final_normalization'].get_weights()
MyGemma.get_layer('Output-Norm').set_weights(fln_weights)
from tqdm import tqdm
for i in tqdm(range(gemma_config['num_layers'])):
    block = layers_dict['decoder_block_'+str(i)]
    attention_name = 'Transformer-%d-MultiHeadSelfAttention' % i
    feed_forward_name = 'Transformer-%d-FeedForward' % i

    MyGemma.get_layer('%s-Norm' % attention_name).set_weights([block.get_weights()[0]])
    MyGemma.get_layer(attention_name).set_weights(block.get_weights()[1:5])
    MyGemma.get_layer('%s-Norm' % feed_forward_name).set_weights([block.get_weights()[5]])
    MyGemma.get_layer(feed_forward_name).set_weights(block.get_weights()[6:])

MyGemma.save_weights(model_name+'/model.weights.h5')
x = keras.ops.arange(1,11)
x = keras.ops.reshape(x,[2,-1])
prompt = {
    # Token ids for "<bos> Keras is".
    "token_ids": x,
    # Use `"padding_mask"` to indicate values that should not be overridden.
    "padding_mask": keras.ops.ones_like(x),
}
print(gemma.predict(prompt))
print(keras.ops.mean(gemma.predict(prompt)-MyGemma.predict(x),-1))



