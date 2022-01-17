from little_dinosaur.download_utils import *
from little_dinosaur.load_pre_model import *
from little_dinosaur.confident_learning import *
from little_dinosaur.nlp_classification_model import *

# from download_utils import *
# from load_pre_model import *
# from confident_learning import *

import fairies as fa
import numpy as np
import os

def eat(

        fileName,        
        true_rate = 0.3,
        k_flod_times = 10,
        test_size = 0.1
    
    ):
    
    """
        置信学习
    """

    pre_training_path = 'pre_model'
    if not os.path.exists(pre_training_path):
        os.makedirs(pre_training_path)

    if not download_verify(pre_training_path):
        download_pretrain_model(pre_training_path)

    output = find_noisy_label_by_confident_learning(
        fileName,
        pre_training_path,
        k_flod_times = k_flod_times,
        test_size = test_size
    )

    if not os.path.exists("log"):
        os.makedirs("log")

    fa.write_npy('log/output.npy',output)

    # 现在是每个模型学习的结果，后面就是主动学习的处理
    # 写到log中

    all = fa.read_json(fileName)

    id2label = {}

    for right in all:
        id2label[right['id']] = int(right['label'])

    res = {}

    for i in output:

        for j in i:
            if j not in res:
                res[j] = []
            for r in  i[j]:
                new = np.array(r)
                
                if new.max() > true_rate :
                    res[j].append(new.argmax(axis=0))

    maybe_wrong = []

    for j in res:
        if len(res[j]) > 1 and len(list(set(res[j]))) ==1:
            if id2label[j]  not in res[j]:
                maybe_wrong.append(j)

    fa.write_json('log/wrong_case.json',maybe_wrong)

def self_learning(
    fileName,
    other_pre_model = False,
    learning_times = 5,
    maxlen = 48,
    batch_size = 96,
    epochs = 20,
    isPair = False,
    model_name = 'bert',
    test_size = 0.4,
    model_path = 'model_self_learning/',
    learning_rate = 3e-4,
    config_path = "",
    checkpoint_path = "",
    dict_path = "",
):
    
    all = fa.read(fileName)
    temp = []

    if not os.path.exists("log"):
        os.makedirs("log")

    for i in range(learning_times):

        train_classification_model(

                fileName,
                trainingFile = fileName,
                other_pre_model = other_pre_model,
                maxlen = maxlen,
                batch_size = batch_size,
                epochs = epochs,
                isPair = isPair,
                model_name = model_name,
                test_size = test_size,
                model_path = model_path,
                learning_rate = learning_rate,
                config_path = config_path,
                checkpoint_path = checkpoint_path,
                dict_path = dict_path,

            )

        model_weight_path = model_path + '/best_model.weights'

        res = predict_classification_model(
                fileName,
                fileName,  
                model_weight_path,
                other_pre_model = other_pre_model,
                maxlen = maxlen,
                batch_size = batch_size,
                isPair = isPair,
                isProbability = False,
                model_name = model_name,
                config_path = config_path,
                checkpoint_path = checkpoint_path,
                dict_path = dict_path,
        )

        for a in all:
            if a['id'] in res:
                if a["label"] == res[a['id']][0]:
                    if a not in temp:
                        temp.append(a)

        fa.write_json('log/right.json',temp)

    all = fa.read(fileName)
    temp = fa.read_json("log/right.json")

    maybe_wrong = []

    for a in all:
        if a not in temp:
            maybe_wrong.append(a)

    fa.write_json('log/wrong_case.py',maybe_wrong)        