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
from tqdm import tqdm


def classification_model(
    config_path,
    checkpoint_path,
    class_num,
    learning_rate=2e-4,
    isElectra=True,
):

    if isElectra:
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            model='electra',
        )
    else:
        bert = build_transformer_model(config_path, checkpoint_path)

    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.output)

    final_output = Dense(class_num, activation='softmax')(output)
    model = Model(bert.inputs, final_output)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate),  
        metrics=['accuracy']
    )

    return model