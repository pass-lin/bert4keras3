import os
os.environ["KERAS_BACKEND"] = "torch"

max_len=30
import keras
keras.config.set_dtype_policy("bfloat16")
from bert4keras3.tokenizers import SpTokenizer
from bert4keras3.models import build_transformer_model

base_path = 'gemma_2b_en/'
config_path = base_path+'config.json'
weights_path = base_path+'model.weights.h5'
dict_path = base_path+'tokenizer.model'

tokenizer = SpTokenizer(base_path+'tokenizer.model', token_start= "<bos>",token_end= "<eos>")

Gemma = build_transformer_model(
    config_path,
    keras_weights_path=weights_path, 
    model='gemma',
    with_lm=True,
    return_keras_model=False,
)

gemma = Gemma.model
gemma.summary()
import numpy as np 

cache_model=Gemma.build_cache_model([max_len],end_token=1,progress_print=True,search_mode='topp',k=0.7)

test_text = '''I want to say '''
tokens2 = tokenizer.encode(test_text)[0][:-1]
#print(tokens2)
tokens2=np.expand_dims(tokens2+[0]*(max_len-len(tokens2)),0)
tokens2 = np.concatenate([tokens2,tokens2],0)
tokens2 = cache_model.predict(tokens2)
print(tokenizer.decode([int(t) for t in tokens2[0]]))
print(tokenizer.decode([int(t) for t in tokens2[1]]))