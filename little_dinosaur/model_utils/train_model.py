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


def train_classification_model(data, save_name,isDrop_noisy=False):

    def evaluate(data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total

    class Evaluator(keras.callbacks.Callback):
        """评估与保存
            """

        def __init__(self):
            self.best_val_acc = 0.

        def on_epoch_end(self, epoch, logs=None):
            val_acc = evaluate(valid_generator)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                model.save_weights(save_name)
            print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
                  (val_acc, self.best_val_acc, 0))

    conf = utils.read_config("model.conf")

    config_path = conf["config_path"]
    checkpoint_path = conf["checkpoint_path"]
    dict_path = conf["dict_path"]
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    maxlen = int(conf["maxlen"])
    batch_size = int(conf["batch_size"])
    epochs = int(conf["epochs"])
    learning_rate = float(conf["learning_rate"])

    # 合理检查
    data = data_utils.add_id(data)

    train_data, test_data, id2label, label2id = data_utils.load_data(
        data)
    class_num = len(id2label)
       
    train_generator = data_generators.classification(train_data, batch_size)
    valid_generator = data_generators.classification(test_data, batch_size)

    model = bert_model.classification_model(config_path,
                                            checkpoint_path,
                                            class_num,
                                            learning_rate=learning_rate)

    evaluator = Evaluator()

    if isDrop_noisy and epochs > 1:

        noisy_data, temp_data = [], []

        model.fit(train_generator.forfit(),
                  steps_per_epoch=len(train_generator),
                  epochs=1,
                  callbacks=[evaluator])

        res = []

        for x_true, y_true in train_generator:

            y_pred_res = model.predict(x_true)
            y_true = y_true[:, 0]

            y_res = y_pred_res.tolist()
            res.extend(y_res)

        for t, r in zip(train_data, res):
            label = t[1]
            # 根据标注的类别计算差异
            if r[label] < (1/class_num/4):
                noisy_data.append(t)
            else:
                temp_data.append(t)

        temp_generator = data_generators.classification(
            temp_data, batch_size)

        model.fit(train_generator.forfit(),
                  steps_per_epoch=len(temp_generator),
                  epochs=epochs - 1,
                  callbacks=[evaluator])

        res = []

        for x_true, y_true in temp_generator:

            y_pred_res = model.predict(x_true)
            y_true = y_true[:, 0]

            y_res = y_pred_res.tolist()
            res.extend(y_res)

        for t, r in zip(temp_data, res):
            label = t[1]
            if r[label] < (1/class_num/3):
                noisy_data.append(t)

        # noisy_data数据格式 [sentence_1, sentence_2, label, noisy_id]
        noisy_ids, output = {}, []
        for i in noisy_data:
            noisy_ids[i[2]] = 0

        for i in data:
            id = i["noisy_id"]
            if id in noisy_ids:
                output.append(i)

        noisy_list = list(noisy_ids)

        keras.backend.clear_session()

        return output, noisy_list, data

    else:

        model.fit(train_generator.forfit(),
                  steps_per_epoch=len(train_generator),
                  epochs=epochs,
                  callbacks=[evaluator])

        res = []

        noisy_data = []

        for x_true, y_true in train_generator:

            y_pred_res = model.predict(x_true)
            y_true = y_true[:, 0]

            y_res = y_pred_res.tolist()
            res.extend(y_res)

        for t, r in zip(train_data, res):
            label = t[1]
            # 根据标注的类别计算差异
            if r[label] < (1/class_num/3):
                noisy_data.append(t)

        noisy_ids, output = {}, []
        for i in noisy_data:
            noisy_ids[i[2]] = 0

        for i in data:
            id = i["noisy_id"]
            if id in noisy_ids:
                output.append(i)

        noisy_list = list(noisy_ids)

        keras.backend.clear_session()
        
        return output, noisy_list, data



def train_classification_model_with_pair(data, save_name, isDrop_noisy=False):

    def evaluate(data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total

    class Evaluator(keras.callbacks.Callback):
        """评估与保存
            """

        def __init__(self):
            self.best_val_acc = 0.

        def on_epoch_end(self, epoch, logs=None):
            val_acc = evaluate(valid_generator)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                model.save_weights(save_name)
            print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
                  (val_acc, self.best_val_acc, 0))

    conf = utils.read_config("model.conf")

    config_path = conf["config_path"]
    checkpoint_path = conf["checkpoint_path"]
    dict_path = conf["dict_path"]
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    maxlen = int(conf["maxlen"])
    batch_size = int(conf["batch_size"])
    epochs = int(conf["epochs"])
    learning_rate = float(conf["learning_rate"])
    # 合理检查

    data = data_utils.add_id(data)

    train_data, test_data, id2label, label2id = data_utils.load_data_pair_texts(
        data)
    class_num = len(id2label)

    train_generator = data_generators.classification_with_pair(
        train_data, batch_size)
    valid_generator = data_generators.classification_with_pair(
        test_data, batch_size)

    model = bert_model.classification_model(config_path,
                                            checkpoint_path,
                                            class_num,
                                            learning_rate=learning_rate)

    evaluator = Evaluator()

    if isDrop_noisy and epochs > 1:

        noisy_data, temp_data = [], []

        model.fit(train_generator.forfit(),
                  steps_per_epoch=len(train_generator),
                  epochs=1,
                  callbacks=[evaluator])

        res = []

        for x_true, y_true in train_generator:

            y_pred_res = model.predict(x_true)
            y_true = y_true[:, 0]

            y_res = y_pred_res.tolist()
            res.extend(y_res)

        for t, r in zip(train_data, res):
            label = t[2]
            # 根据标注的类别计算差异
            if r[label] < (1/class_num/4):
                noisy_data.append(t)
            else:
                temp_data.append(t)

        temp_generator = data_generators.classification_with_pair(
            temp_data, batch_size)

        model.fit(train_generator.forfit(),
                  steps_per_epoch=len(temp_generator),
                  epochs=epochs - 1,
                  callbacks=[evaluator])

        res = []

        for x_true, y_true in temp_generator:

            y_pred_res = model.predict(x_true)
            y_true = y_true[:, 0]

            y_res = y_pred_res.tolist()
            res.extend(y_res)

        for t, r in zip(temp_data, res):
            label = t[2]
            if r[label] < (1/class_num/3):
                noisy_data.append(t)

        # noisy_data数据格式 [sentence_1, sentence_2, label, noisy_id]
        noisy_ids, output = {}, []
        for i in noisy_data:
            noisy_ids[i[3]] = 0

        for i in data:
            id = i["noisy_id"]
            if id in noisy_ids:
                output.append(i)

        noisy_list = list(noisy_ids)

        keras.backend.clear_session()

        return output, noisy_list, data

    else:

        model.fit(train_generator.forfit(),
                  steps_per_epoch=len(train_generator),
                  epochs=epochs,
                  callbacks=[evaluator])

        res = []

        noisy_data = []

        for x_true, y_true in train_generator:

            y_pred_res = model.predict(x_true)
            y_true = y_true[:, 0]

            y_res = y_pred_res.tolist()
            res.extend(y_res)

        for t, r in zip(train_data, res):
            label = t[2]
            # 根据标注的类别计算差异
            if r[label] < (1/class_num/3):
                noisy_data.append(t)

        noisy_ids, output = {}, []
        for i in noisy_data:
            noisy_ids[i[3]] = 0

        for i in data:
            id = i["noisy_id"]
            if id in noisy_ids:
                output.append(i)

        noisy_list = list(noisy_ids)

        keras.backend.clear_session()
        
        return output, noisy_list, data


if __name__ == '__main__':

    pass
