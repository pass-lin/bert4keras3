config={
  "type_vocab_size": 0, 
  
  
}

import os
os.environ['KAGGLE_USERNAME'] = 'passlin'
os.environ['KAGGLE_KEY'] = '848acf174f9616e8eeae54691d758f93'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size=14
epochs = 15
os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras_nlp
from bert4keras3.models import build_transformer_model
from bert4keras3.snippets import sequence_padding
keras.config.set_dtype_policy("bfloat16")
model_name = "gemma2_9b_en"
import torch
with torch.no_grad():
    try:
        os.makedirs(model_name)
    except:
        pass
    gemma =  keras_nlp.models.GemmaCausalLM.from_preset(model_name)
    from keras import ops
    backbone = gemma.get_layer('gemma_backbone')
    layers_dict = {}
    for layer in backbone.layers:
        if layer.weights!=[]:
            layers_dict[layer.name]=layer
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
    config[ "use_post_ffw_norm"]=backbone.use_post_ffw_norm
    config[ "use_post_attention_norm"] = backbone.use_post_attention_norm
    config[ "logit_soft_cap"] = layers_dict['decoder_block_'+str(0)].attention.logit_soft_cap
    config[ "use_sliding_window_attention"] = layers_dict['decoder_block_'+str(0)].attention.use_sliding_window_attention
    config[ "sliding_window_size"] = layers_dict['decoder_block_'+str(0)].attention.sliding_window_size
    config[ "query_head_dim_normalize"] = layers_dict['decoder_block_'+str(0)].attention.query_head_dim_normalize
    hidden_dim=config[ "hidden_size"]
    import json
    with open(model_name+'/config.json', 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    gemma.eval()
    self = build_transformer_model(
            config_path=model_name+'/config.json',
            model='gemma2',
            return_keras_model=False,
            with_lm='linear',
        )
    MyGemma= self.model
    MyGemma.eval()
    gemma.summary()


    

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
        
        
        MyGemma.get_layer('%s-Norm' % attention_name).set_weights([block.weights[0]])
        MyGemma.get_layer('%s-Norm-post' % attention_name).set_weights([block.weights[1]])
        MyGemma.get_layer(attention_name).set_weights(block.weights[2:6])

        MyGemma.get_layer('%s-Norm' % feed_forward_name).set_weights([block.weights[6]])
        
        MyGemma.get_layer('%s-Norm-post' % feed_forward_name).set_weights([block.weights[7]])
        MyGemma.get_layer(feed_forward_name).set_weights(block.weights[8:])
            

    import numpy as np 
    MyGemma.save_weights(model_name+'/model.weights.h5')
    print('saving')
    x = np.random.randint(1,100000,[3,128])
    x1 = gemma([np.ones_like(x),x])
    x2 = MyGemma(x)
    print('-'*20)
    print(keras.ops.mean(keras.ops.abs(x1-x2)))
    print(keras.ops.max(keras.ops.abs(x1-x2)))