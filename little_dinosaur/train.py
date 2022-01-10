from little_dinosaur import utils
from little_dinosaur import load_pre_model
from little_dinosaur import model_nlp

def train_classification_model(

        fileName,
        pre_training_path,
        test_size = 0.1,
        maxlen = 48,
        batch_size = 96
    ):

    pre_model = load_pre_model.load_pre_model(pre_training_path)
    
    for pre in pre_model:

        if pre == "electra_180g_small": 

            model_name = 'electra'

            if pre == 'albert':
                model_name = 'albert' 
            elif 'electra' in pre :
                model_name = 'electra'
            else:
                model_name = 'bert'

            k_flod_times = 1

            res  = model_nlp.classification_model(
                fileName,        
                pre_model[pre]['config_path'],
                pre_model[pre]['checkpoint_path'],
                pre_model[pre]['dict_path'],
                isPair = False,
                maxlen = maxlen,
                batch_size = batch_size,
                model_name = model_name,
                k_flod_times = k_flod_times,
                test_size = test_size
            )


