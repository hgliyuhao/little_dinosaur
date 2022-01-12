from little_dinosaur.download_utils import *
import os

def load_pre_model(path):
    
    pre_model = {}
    
    # dir_path = os.path.dirname(os.path.abspath(__file__))
    pre_path = os.path.join(path,'pre_model')
    pre_model['albert'] = {}
    albert_pre_path = os.path.join(pre_path,'albert_tiny_zh_google')

    albert_config_path = os.path.join(albert_pre_path,'albert_config_tiny_g.json')
    albert_checkpoint_path = os.path.join(albert_pre_path,'albert_model.ckpt')
    albert_dict_path = os.path.join(albert_pre_path,'vocab.txt')

    pre_model['albert']['config_path'] = albert_config_path
    pre_model['albert']['checkpoint_path'] = albert_checkpoint_path
    pre_model['albert']['dict_path'] = albert_dict_path
    
    pre_model['electra_small'] = {}
    electra_small_pre_path = os.path.join(pre_path,'electra-small')

    electra_small_config_path = os.path.join(electra_small_pre_path,'bert_config_tiny.json')
    electra_small_checkpoint_path = os.path.join(electra_small_pre_path,'electra_small')
    electra_small_dict_path = os.path.join(electra_small_pre_path,'vocab.txt')

    pre_model['electra_small']['config_path'] = electra_small_config_path
    pre_model['electra_small']['checkpoint_path'] = electra_small_checkpoint_path
    pre_model['electra_small']['dict_path'] = electra_small_dict_path

    pre_model['electra_180g_small'] = {}
    electra_180g_small_pre_path = os.path.join(pre_path,'electra_180g_small')

    electra_180g_small_config_path = os.path.join(electra_180g_small_pre_path,'small_discriminator_config.json')
    electra_180g_small_checkpoint_path = os.path.join(electra_180g_small_pre_path,'electra_180g_small.ckpt')
    electra_180g_small_dict_path = os.path.join(electra_180g_small_pre_path,'vocab.txt')

    pre_model['electra_180g_small']['config_path'] = electra_180g_small_config_path
    pre_model['electra_180g_small']['checkpoint_path'] = electra_180g_small_checkpoint_path
    pre_model['electra_180g_small']['dict_path'] = electra_180g_small_dict_path

    return pre_model

def get_config_path(

    other_pre_model,
    config_path,    
    checkpoint_path,
    dict_path,
    pre_training_path = 'pre_model',
    model = 'electra_180g_small',
    model_name = 'bert'

):  
    pre_model = load_pre_model(pre_training_path)
    model_list = ['albert','electra_small','electra_180g_small']
    if model not in model_list:
            raise ValueError(
                "Model must be albert ,electra_small or electra_180g_small"
    )

    if other_pre_model:
        
        if config_path != '' and checkpoint_path != '' and dict_path != '':
            pass
        else:
            raise ValueError(
                "If use other_pre_model, you may need to set config_path,checkpoint_path,dict_path "
            )

    else:

        pre_training_path = 'pre_model'
        if not os.path.exists(pre_training_path):
            os.makedirs(pre_training_path)

        if not download_verify(pre_training_path):
            download_pretrain_model(pre_training_path)

        model_list = ['albert','electra_small','electra_180g_small']
        if model not in model_list:
            raise ValueError(
                "model must be albert ,electra_small or electra_180g_small"
            )

        for pre in pre_model:
            if pre == model: 
                if pre == 'albert':
                    model_name = 'albert' 
                elif 'electra' in pre :
                    model_name = 'electra'
                else:
                    model_name = 'bert'
                
                config_path = pre_model[pre]['config_path']
                checkpoint_path = pre_model[pre]['checkpoint_path']
                dict_path = pre_model[pre]['dict_path']

    return config_path,checkpoint_path,dict_path,pre_training_path,model_name