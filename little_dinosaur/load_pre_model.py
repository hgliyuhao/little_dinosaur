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