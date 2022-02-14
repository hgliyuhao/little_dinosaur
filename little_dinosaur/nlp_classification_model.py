from little_dinosaur import utils
from little_dinosaur import load_pre_model

from keras.layers import *
from bert4keras.backend import keras, set_gelu, K, search_layer
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
import numpy as np
import fairies as fa
import os
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

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

def classification_model(
        model_name,
        config_path,
        checkpoint_path,
        num_class,
        learning_rate = 2e-5
    ):
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
            optimizer=Adam(learning_rate),  # 用足够小的学习率
            metrics=['accuracy'],            
        )

        return model

def train_classification_model(

        fileName,
        # pre_training_path, 
        trainingFile = '',
        vaild_fileName = '',
        other_pre_model = False,
        maxlen = 48,
        batch_size = 96,
        epochs = 10,
        learning_rate = 2e-5,
        isPair = False,
        isDynamicLr = False,
        isBalanceClassWeight = False,
        model_name = 'bert',
        test_size = 0.2,
        model_path = 'model/',
        config_path = '',
        checkpoint_path = '',
        dict_path = '',

    ):
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    res_dict = {}
    
    # TODO
    # 先用模型训练一次 选取最好的epoch
    # 判断数据是否合理
    
    # 得到num_class
    
    if trainingFile == "":
        trainingFile = fileName

    all_data = fa.read_json(trainingFile)
    all_class = set()

    labels,weight_labels,weight_classes = [],[],[]

    for i in all_data:
        all_class.add(i["label"])
        labels.append(i["label"])

    num_class = len(all_class)

    id2label = dict(enumerate(sorted(list(all_class))))
    label2id = {j: i for i, j in id2label.items()}
    
    # compute_weight
    for id in id2label:
        weight_classes.append(id)

    for l in labels:
        weight_labels.append(label2id[l])

    class_weight = 'balanced'
    weight = compute_class_weight(class_weight, weight_classes, weight_labels)
    weight = dict(enumerate(weight))

    for l in labels:
        weight_labels.append(label2id[l])

    config_path,checkpoint_path,dict_path,pre_training_path,model_name = load_pre_model.get_config_path(
        other_pre_model,
        config_path,    
        checkpoint_path,
        dict_path,
    )

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
                batch_labels.append([label2id[label]])

                if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    if vaild_fileName == "":
        train_data,valid_data = utils.random_split_data(fileName,test_size)
        fa.write_json(model_path +'train.json',train_data,isLine=True)
        fa.write_json(model_path +'valid.json',valid_data,isLine=True)
    else:
        train_data = fa.read_json(fileName)
        valid_data = fa.read_json(vaild_fileName)

    train_data = read_data_by_data(train_data)
    valid_data = read_data_by_data(valid_data)

    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size)
        
    model = classification_model(model_name,config_path,checkpoint_path,num_class,learning_rate)

    def evaluate(data):
        total, right = 1., 1.
        for x_true, y_true in data:
            res = model.predict(x_true)
            y_pred = res.argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        
        return right / total

    class Evaluator(keras.callbacks.Callback):
        def __init__(self):
            self.best_val_acc = 0.
            model.save_weights(model_path + 'last_model.weights')

        def on_epoch_end(self, epoch, logs=None):
            if epoch == 0:
                self.best_val_acc = 0.
            val_acc = evaluate(valid_generator)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                model.save_weights(model_path + 'best_model.weights')
            print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n'
                % (val_acc, self.best_val_acc, 0))

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience= 2, mode='auto')  
    evaluator = Evaluator()

    if isDynamicLr:

        if isBalanceClassWeight:
            model.fit(
                train_generator.forfit(),
                steps_per_epoch=len(train_generator),
                epochs = epochs,
                class_weight = weight,
                callbacks=[evaluator]
            )
        else:    
            model.fit(
                train_generator.forfit(),
                steps_per_epoch=len(train_generator),
                epochs = epochs,
                callbacks=[evaluator,reduce_lr]
            )   
    else:
        if isBalanceClassWeight:
            model.fit(
                train_generator.forfit(),
                steps_per_epoch=len(train_generator),
                epochs = epochs,
                class_weight = weight,
                callbacks=[evaluator]
            )
        else:
            model.fit(
                train_generator.forfit(),
                steps_per_epoch=len(train_generator),
                epochs = epochs,
                callbacks=[evaluator]
            )

    keras.backend.clear_session()

def predict_classification_model(
        fileName,  
        # pre_training_path,
        trainingFile,
        model_weight_path,
        other_pre_model = False,
        learning_rate = 2e-5,
        maxlen = 48,
        batch_size = 96,
        isPair = False,
        isProbability = False,
        model_name = 'bert',
        config_path = '',
        checkpoint_path = '',
        dict_path = '',   
):

    all_data = fa.read_json(trainingFile)
    all_class = set()

    for i in all_data:
        all_class.add(i["label"])

    num_class = len(all_class)
    id2label = dict(enumerate(sorted(list(all_class))))
    label2id = {j: i for i, j in id2label.items()}
    
    config_path,checkpoint_path,dict_path,pre_training_path,model_name = load_pre_model.get_config_path(
        other_pre_model,
        config_path,    
        checkpoint_path,
        dict_path,
        # pre_training_path,
    )

    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    class data_generator(DataGenerator):
    
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
                batch_labels.append([label2id[label]])

                if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
   
    test_data = read_data(fileName)
    test_generator = data_generator(test_data, batch_size)
    model = classification_model(model_name,config_path,checkpoint_path,num_class,learning_rate)
    
    model.load_weights(model_weight_path)

    res,res_dict = [],{}

    if isProbability:
        for x_true, y_true in test_generator:
            res.extend(model.predict(x_true).tolist())
    else:        
        for x_true, y_true in test_generator:
            res.extend(model.predict(x_true).argmax(axis=1))        

    for i,valid in enumerate(test_data):
        if valid[2] not in res_dict:
            res_dict[valid[2]] = [] 
        res_dict[valid[2]].append(id2label[res[i]])
        
    keras.backend.clear_session()

    return res_dict
