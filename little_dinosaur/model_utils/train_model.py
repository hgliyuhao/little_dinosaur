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


def train_classification_model(data,
                               save_name,
                               id2label,
                               label2id,
                               isPair=False):

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

    config_path, checkpoint_path, dict_path, maxlen, batch_size, epochs, learning_rate = utils.read_config(
        "model.conf")
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    # 合理检查
    train_data, test_data = data_utils.load_data(data,
                                                 id2label,
                                                 label2id,
                                                 isPair=isPair)

    class_num = len(id2label)

    if isPair:

        train_generator = data_generators.classification_with_pair(
            train_data, batch_size)
        valid_generator = data_generators.classification_with_pair(
            test_data, batch_size)

    else:

        train_generator = data_generators.classification(
            train_data, batch_size)
        valid_generator = data_generators.classification(test_data, batch_size)

    model = bert_model.classification_model(config_path,
                                            checkpoint_path,
                                            class_num,
                                            learning_rate=learning_rate)

    evaluator = Evaluator()

    model.fit(train_generator.forfit(),
              steps_per_epoch=len(train_generator),
              epochs=epochs,
              callbacks=[evaluator])

    keras.backend.clear_session()

    return model

if __name__ == '__main__':

    pass
