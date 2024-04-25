import os
os.environ["KERAS_BACKEND"] = "torch"

max_len=300
import keras
keras.config.set_dtype_policy("bfloat16")
from transformers import AutoTokenizer
from bert4keras3.models import build_transformer_model


base_path = 'Yi-6B/'
config_path = base_path+'config.json'
weights_path = base_path+'model.weights.h5'
dict_path = base_path+'Yi_tokenizer'

tokenizer = AutoTokenizer.from_pretrained(dict_path)

Qwen = build_transformer_model(
    config_path,
    keras_weights_path=weights_path, 
    model='qwen',
    with_lm=True,
    return_keras_model=False,
)

gemma = Qwen.model
gemma.summary()
import numpy as np 
#yi的end_token是2，llama3是128001，具体自己看hf的config
cache_model=Qwen.build_cache_model([max_len],end_token=2,progress_print=True,search_mode='topp',k=0.7)

test_text = '''楼房大小的血睛鬃毛狮，力大无穷的紫睛金毛猿，毁天灭地的九头蛇皇，携带着毁灭雷电的恐怖雷龙…… '''
tokens2 = tokenizer(test_text)['input_ids']
print(tokens2)
tokens2=np.expand_dims(tokens2+[0]*(max_len-len(tokens2)),0)
tokens2 = np.concatenate([tokens2,tokens2],0)
tokens2 = cache_model.predict(tokens2)
print(tokenizer.decode(tokens2[0]))
print(tokenizer.decode(tokens2[1]))