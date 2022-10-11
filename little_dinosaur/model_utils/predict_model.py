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


def predict_classification_model(train_data, test_data, weights_name):

    config_path,checkpoint_path,dict_path,maxlen,batch_size,epochs,learning_rate = utils.read_config("model.conf")
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    # 合理检查
    test, id2label, label2id = data_utils.load_test_data(
        train_data, test_data)

    class_num = len(id2label)

    test_generator = data_generators.classification(test, batch_size)

    model = bert_model.classification_model(
        config_path,
        checkpoint_path,
        class_num,
    )

    model.load_weights(weights_name)

    res = []

    for x_true, y_true in test_generator:

        y_pred_res = model.predict(x_true)
        y_pred = y_pred_res.argmax(axis=1)
        y_true = y_true[:, 0]

        y_res = y_pred_res.tolist()
        res.extend(y_res)

    for t,r in zip(test_data,res):
        t["score"] = r

    keras.backend.clear_session()

    return test_data

def predict_classification_model_with_pair(train_data, test_data, weights_name):

    config_path,checkpoint_path,dict_path,maxlen,batch_size,epochs,learning_rate = utils.read_config("model.conf")
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    # 合理检查
    test, id2label, label2id = data_utils.load_test_data_pair_texts(
        train_data, test_data)

    class_num = len(id2label)

    test_generator = data_generators.classification_with_pair(test, batch_size)

    model = bert_model.classification_model(
        config_path,
        checkpoint_path,
        class_num,
    )

    model.load_weights(weights_name)

    res = []

    for x_true, y_true in test_generator:

        y_pred_res = model.predict(x_true)
        y_pred = y_pred_res.argmax(axis=1)
        y_true = y_true[:, 0]

        y_res = y_pred_res.tolist()
        res.extend(y_res)

    for t,r in zip(test_data,res):
        t["score"] = r

    keras.backend.clear_session()

    return test_data


if __name__ == '__main__':
    pass
