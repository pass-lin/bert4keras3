import os
config={
  "type_vocab_size": 0, 
  'hidden_act':'gelu'
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras_nlp
from bert4keras3.models import build_transformer_model
from bert4keras3.snippets import sequence_padding
from keras import ops
model_name = "Deberta_v3_base_multi"

try:
    os.makedirs(model_name)
except:
    pass

import shutil
import os

# 源文件路径
source_file_path = './keras_nlp_weights/deberta_v3-keras-%s-v2/assets/tokenizer/vocabulary.spm'%model_name.lower()

shutil.move(source_file_path, model_name+'/')
model = keras_nlp.models.DebertaV3MaskedLM.from_preset('./keras_nlp_weights/deberta_v3-keras-%s-v2'%model_name.lower(),preprocessor=None)
model.eval()

backbone = model.layers[2]
deberta_config = backbone.get_config()

config[ "vocab_size"]=deberta_config['vocabulary_size']
config[ "num_hidden_layers"]=deberta_config['num_layers']
config[ "num_attention_heads"]=deberta_config['num_heads']
config[ "hidden_size"]=deberta_config['hidden_dim']
config[ "intermediate_size"]=deberta_config['intermediate_dim']
config[ "attention_probs_dropout_prob"]=deberta_config['dropout']
config[ "dropout_rate"]=deberta_config['dropout']
config[ "max_position"]=deberta_config['max_sequence_length']
config[ "bucket_size"]=deberta_config['bucket_size']
import json
with open(model_name+'/config.json', 'w') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)

mydeberta = build_transformer_model(
        config_path=model_name+'/config.json',
        model='deberta',
        return_keras_model=True,
        with_mlm='linear',
    )

mydeberta.get_layer('Embedding-Token').set_weights(backbone.layers[1].get_weights())
mydeberta.get_layer('Embedding-Norm').set_weights(backbone.layers[2].get_weights())
mydeberta.get_layer('Embedding-Deberta-Position').set_weights(backbone.layers[5].get_weights())
mydeberta.eval()
for index in range(config[ "num_hidden_layers"]):
    layers = backbone.layers[6+index]
    attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
    feed_forward_name = 'Transformer-%d-FeedForward' % index

    mydeberta.get_layer(attention_name).set_weights(layers._self_attention_layer.get_weights())
    mydeberta.get_layer(feed_forward_name).set_weights(layers._feedforward_intermediate_dense.get_weights()+layers._feedforward_output_dense.get_weights())
    mydeberta.get_layer('%s-Norm' % attention_name).set_weights(layers._self_attention_layer_norm.get_weights())
    mydeberta.get_layer('%s-Norm' % feed_forward_name).set_weights(layers._feedforward_layer_norm.get_weights())

import numpy as np
from bert4keras3.tokenizers import SpTokenizer
tokenizer = SpTokenizer(model_name+'/vocabulary.spm')
mydeberta.layers[-1].set_weights(model.layers[-1].get_weights())

features = {
    "token_ids": np.array([[1, 2, 3, 4, 3, 6, 7, 8,0]] * 2),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1,0]] * 2),
    "mask_positions": np.array([[2, 4]] * 2),
}
z1 = model.predict(features)
z2 = mydeberta.predict([features["token_ids"],
                        features["mask_positions"]])
print(np.sum(z1-z2))
mydeberta.save_weights(model_name+'/model.weights.h5')