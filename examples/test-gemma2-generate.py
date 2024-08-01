import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
max_len=30
import keras
keras.config.set_dtype_policy("bfloat16")
from bert4keras3.tokenizers import SpTokenizer
from bert4keras3.models import build_transformer_model
from transformers import AutoTokenizer
base_path = 'gemma2_9b_en/'
config_path = base_path+'config.json'
weights_path = base_path+'model.weights.h5'
dict_path = base_path+'tokenizer'

tokenizer = AutoTokenizer.from_pretrained(dict_path)

Gemma = build_transformer_model(
    config_path,
    keras_weights_path=weights_path, 
    model='gemma2',
    with_lm=True,
    return_keras_model=False,
)

gemma = Gemma.model
import numpy as np 

cache_model=Gemma.build_cache_model([max_len],end_token=1,progress_print=True,search_mode='topp',k=0.7)

test_text = ''' keras is a deep-learning '''
tokens2 = tokenizer.encode(test_text)
#print(tokens2)
print(len(tokens2))
tokens2=np.expand_dims(tokens2+[0]*(max_len-len(tokens2)),0)
tokens2 = np.concatenate([tokens2,tokens2],0)
tokens2 = cache_model.predict(tokens2)
print(tokenizer.decode([int(t) for t in tokens2[0]]))
print(tokenizer.decode([int(t) for t in tokens2[1]]))