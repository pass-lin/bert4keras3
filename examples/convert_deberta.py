#获取deberta的embedding
import os
os.environ["KERAS_BACKEND"] = "torch"
model_name = "Deberta_v3_base_multi"
from bert4keras3.tokenizers import SpTokenizer
import numpy as np
from bert4keras3.models import build_transformer_model
tokenizer = SpTokenizer(model_name+'/vocabulary.spm')
mydeberta = build_transformer_model(
        config_path=model_name+'/config.json',
        keras_weights_path=model_name+'/model.weights.h5',
        model='deberta',
        return_keras_model=True,
        with_mlm=False,
    )
text = "Always get the best performance for your models. In our benchmarks, we found that JAX typically delivers the best training and inference performance on GPU, TPU, and CPU – but results vary from model to model, as non-XLA TensorFlow is occasionally faster on GPU. The ability to dynamically select the backend that will deliver the best performance for your model without having to change anything to your code means you're always guaranteed to train and serve with the highest achievable'"
x = np.reshape(tokenizer.encode(text)[0],[1,-1])
w = mydeberta.predict(x)
print(w)
