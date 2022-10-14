import fairies as fa
from model_utils import data_utils
from main import *

data = fa.read("data/example.json")
data = data_utils.convert_data_to_train(data)

pair_data = fa.read("data/example_pair.json")
pair_data = data_utils.convert_data_to_train(pair_data)

data = data[:500]

pair_data = pair_data[:500]
model_save_name = "model/test.weights"

train_classification_model(data, model_save_name)
predict_classification_model(data, model_save_name, data)

train_classification_model(pair_data, model_save_name, isPair=True)
predict_classification_model(pair_data,
                             model_save_name,
                             pair_data,
                             isPair=True)


find_noisy_label_drop(data,
                      model_save_name,
                      isPair=False,
                      times=2,
                      isDrop_noisy=True)

find_noisy_label_drop(
    data,
    model_save_name,
    isPair=False,
    times=2,
)

find_noisy_label_drop(pair_data,
                      model_save_name,
                      isPair=True,
                      times=2,
                      isDrop_noisy=True)

find_noisy_label_drop(
    pair_data,
    model_save_name,
    isPair=True,
    times=2,
)
