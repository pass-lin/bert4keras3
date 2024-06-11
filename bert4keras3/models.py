#! -*- coding: utf-8 -*-
# 主要模型



from bert4keras3.transformers import *
from bert4keras3.Models.Berts import *
from bert4keras3.Models.Alberts import *
from bert4keras3.Models.Roformers import *
from bert4keras3.Models.GPTs import *
from bert4keras3.Models.T5models import *
from bert4keras3.Models.GPTs import *
from bert4keras3.Models.GAUs import *
from bert4keras3.Models.LLamas import *
def extend_with_language_model(BaseModel):
    """添加下三角的Attention Mask（语言模型用）
    """
    class LanguageModel(LM_Mask, BaseModel):
        """带下三角Attention Mask的派生模型
        """
        def __init__(self, *args, **kwargs):
            super(LanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True

    return LanguageModel


def extend_with_unified_language_model(BaseModel):
    """添加UniLM的Attention Mask（Seq2Seq模型用）
    """
    class UnifiedLanguageModel(UniLM_Mask, BaseModel):
        """带UniLM的Attention Mask的派生模型
        UniLM: https://arxiv.org/abs/1905.03197
        """
        def __init__(self, *args, **kwargs):
            super(UnifiedLanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True

    return UnifiedLanguageModel




def build_transformer_model(
    config_path=None,
    checkpoint_path=None,
    model='bert',
    application='encoder',
    return_keras_model=True,
    keras_weights_path=None,
    **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'attention_dropout_rate' not in configs:
        configs['attention_dropout_rate'] = configs.get(
            'attention_probs_dropout_prob'
        )
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)

    models = {
        'bert': BERT,
        'albert': ALBERT,
        'albert_unshared': ALBERT_Unshared,
        'roberta': BERT,
        'nezha': NEZHA,
        'roformer': RoFormer,
        'roformer_v2': RoFormerV2,
        'electra': ELECTRA,
        'gau':GAU_alpha,
        'gpt': GPT,
        'gpt2': GPT2,
        'gpt2_ml': GPT2_ML,
        't5': T5,
        't5_encoder': T5_Encoder,
        't5_decoder': T5_Decoder,
        't5.1.0': T5,
        't5.1.0_encoder': T5_Encoder,
        't5.1.0_decoder': T5_Decoder,
        't5.1.1': T5,
        't5.1.1_encoder': T5_Encoder,
        't5.1.1_decoder': T5_Decoder,
        'mt5.1.1': T5,
        'mt5.1.1_encoder': T5_Encoder,
        'mt5.1.1_decoder': T5_Decoder,
        'gemma':Gemma,
        'llama':Llama, 
        'qwen':Llama, 
        'yi':Llama, 
        'misakat5':MisakaT5,
    }

    if is_string(model):
        model = model.lower()
        MODEL = models[model]
        if model.startswith('t5.1.1'):
            configs['version'] = 't5.1.1'
        elif model.startswith('mt5.1.1'):
            configs['version'] = 'mt5.1.1'
    else:
        MODEL = model

    application = application.lower()
    if application in ['lm', 'unilm'] and model in ['electra', 't5']:
        raise ValueError(
            '"%s" model can not be used as "%s" application.\n' %
            (model, application)
        )

    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unified_language_model(MODEL)

    transformer = MODEL(**configs)
    transformer.build(**configs)
    if keras.__version__>'3.0':
        #keras3不知道为什么attention需要走一次前向才能初始化
        inputs=[]
        for modelin in transformer.model.inputs: 
            shape=keras.ops.shape(modelin)
            shape=[1 if t==None else t for t in shape]
            try:
                shape[0] = len(keras.distribution.list_devices())
            except:
                pass
            inputs.append(np.ones(shape,modelin.dtype))
        transformer.model.predict(inputs,verbose=3)
        if keras_weights_path is not None:
            transformer.model.load_weights(keras_weights_path, skip_mismatch=True)
        if lora_model:
            
            def enable_lora(t):
                if isinstance(t,keras.layers.Embedding) :
                    t.enable_lora(configs['attention_head_size']*2)
                elif isinstance(t,keras.layers.Dense):
                    t.enable_lora(configs['attention_head_size'])
                else:
                    return True
                return False
                    
            for layer in transformer.model.layers:
                if 'norm' in layer.name.lower():
                    continue
                flag = enable_lora(layer)
                if flag:
                    layer.trainable=False
                for kid in dir (layer):
                    t = getattr(layer,kid)
                    enable_lora(t)
                    if flag:
                        layer.trainable=False
    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer.model
    else:
        return transformer
