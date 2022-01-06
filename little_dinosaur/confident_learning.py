from keras.layers import *
from bert4keras.backend import keras, set_gelu, K, search_layer
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
import random
import numpy as np
import fairies as fa
import utils

# from little_dinosaur import load_pre_model
import load_pre_model
set_gelu('tanh')  # 切换gelu版本
    
def read_data(fileName):

    res = []

    all_data = fa.read_json(fileName)
    
    for i in all_data:
        
        label = i["label"]
        sentence = i["sentence"]
        id = i["id"]
        
        res.append([sentence,label,id])
        
    return res   

def read_data_by_data(all_data):

    res = []

    for i in all_data:
        
        label = i["label"]
        sentence = i["sentence"]
        id = i["id"]
        
        res.append([sentence,label,id])
        
    return res         

def confident_learning(

        fileName,        
        config_path,
        checkpoint_path,
        dict_path,
        maxlen = 48,
        batch_size = 96,
        isPair = False,
        model_name = 'bert',
        k_flod_times = 10,
        test_size = 0.2

    ):
    
    res_dict = {}
    
    # TODO
    # 先用模型训练一次 选取最好的epoch
    # 判断数据是否合理
    
    # 得到num_class
    
    all_data = fa.read_json(fileName)
    all_class = set()

    for i in all_data:
        all_class.add(i["label"])

    num_class = len(all_class)

    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    class data_generator(DataGenerator):
    
        """
            数据生成器
        """

        def __iter__(self, random=False):
            idxs = list(range(len(self.data)))
            if random:
                np.random.shuffle(idxs)
            batch_token_ids, batch_segment_ids, batch_labels = [], [], []
            for i in idxs:

                data = self.data[i]

                if not isPair:
                    text1 = data[0]
                    label = data[1]
                    token_ids, segment_ids = tokenizer.encode(text1,maxlen=maxlen)

                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([int(label)])
                if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    for i in range(int(k_flod_times)):

        train_data,valid_data = utils.random_split_data(fileName,test_size)
        train_data = read_data_by_data(train_data)
        valid_data = read_data_by_data(valid_data)

        train_generator = data_generator(train_data, batch_size)
        valid_generator = data_generator(valid_data, batch_size)
        
        # 加载预训练模型
        if model_name == 'electra':

            bert = build_transformer_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                model='electra',
                return_keras_model=False,

            )  # 建立模型，加载权重
    
        elif model_name == 'albert' :
            bert = build_transformer_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                model='albert',
                return_keras_model=False,
            )

        else:

            bert = build_transformer_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                return_keras_model=False,

            )  # 建立模型，加载权重


        output = Lambda(lambda x: x[:, 0],
                        name='CLS-token')(bert.model.output)
        output = Dense(units=num_class,
                        activation='softmax',
                    kernel_initializer=bert.initializer)(output)

        model = keras.models.Model(bert.model.input, output)
        model.compile(

            loss='sparse_categorical_crossentropy',
            optimizer=Adam(2e-5),  # 用足够小的学习率
            metrics=['accuracy'],
            
        )

        def evaluate(data,type,epoch):
            output = []
            total, right = 0., 0.
            for x_true, y_true in data:
                res = model.predict(x_true)
                output.extend(res)
                y_pred = res.argmax(axis=1)
                y_true = y_true[:, 0]
                total += len(y_true)
                right += (y_true == y_pred).sum()
        
            return right / total

        def predict(data):
            res = []
            for x_true, y_true in data:
                res.extend(model.predict(x_true).tolist())
            return res
        
        class Evaluator(keras.callbacks.Callback):
            def __init__(self):
                self.best_val_acc = 0.
                model.save_weights('best_model.weights')

            def on_epoch_end(self, epoch, logs=None):
                if epoch == 0:
                    self.best_val_acc = 0.
                val_acc = evaluate(valid_generator,'valid',epoch)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    model.save_weights('best_model.weights')
                print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n'
                    % (val_acc, self.best_val_acc, 0))
        
        evaluator = Evaluator()

        model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=10,
            callbacks=[evaluator]
        )

        model.load_weights('best_model.weights')
        
        res = predict(valid_generator)

        for i,valid in enumerate(valid_data):
            if valid[2] not in res_dict:
                res_dict[valid[2]] = [] 
            res_dict[valid[2]].append(res[i])
        
        keras.backend.clear_session()

    return res_dict    

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

        res  = confident_learning(
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
        
