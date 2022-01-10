from little_dinosaur import utils
from little_dinosaur import load_pre_model
from little_dinosaur import model_nlp

def find_noisy_label_by_confident_learning(

        fileName,
        pre_training_path,
        k_flod_times = 10,
        test_size = 0.2,
        maxlen = 48,
        batch_size = 96
    ):

    pre_model = load_pre_model.load_pre_model(pre_training_path)
    output = []
    
    for pre in pre_model:

        if pre == 'albert':
            model_name = 'albert' 
        elif 'electra' in pre :
            model_name = 'electra'
        else:
            model_name = 'bert'   

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
        output.append(res)
    
    return output 
        
