import numpy as np
from keras.layers import *
from keras.models import *
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
import fairies as fa
from tqdm import tqdm
import os
import time

from model_utils import utils
from model_utils import data_utils
from model_utils import data_generators
from model_utils import bert_model


def predict_classification_model(test_data,
                                 weights_name,
                                 id2label,
                                 label2id,
                                 isPair=False):

    config_path, checkpoint_path, dict_path, maxlen, batch_size, epochs, learning_rate = utils.read_config(
        "model.conf")
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    # 合理检查
    test = data_utils.load_test_data(test_data,id2label, label2id, isPair=isPair)

    class_num = len(id2label)

    if isPair:
        test_generator = data_generators.classification_with_pair(test, batch_size)
    else:
        test_generator = data_generators.classification(test, batch_size)    

    model = bert_model.classification_model(
        config_path,
        checkpoint_path,
        class_num,
    )

    model.load_weights(weights_name)

    res = []
    loss = []

    for x_true, y_true in test_generator:

        y_pred_res = model.predict(x_true)
        y_pred = y_pred_res.argmax(axis=1)
        y_true = y_true[:, 0]

        y_res = y_pred.tolist()
        res.extend(y_res)

        for i,score in enumerate(y_pred_res):
            loss.append(score[y_true[i]])

    for num in range(len(test_data)):
        test_data[num]["pre_label"] = id2label[res[num]]
        test_data[num]["loss"] = float(loss[num])

    keras.backend.clear_session()

    return test_data

if __name__ == '__main__':
    pass
