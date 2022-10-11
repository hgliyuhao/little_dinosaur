import fairies as fa
from model_utils import train_model
from model_utils import predict_model
from model_utils import data_utils
from model_utils import find_noisy_labels


def train_classification_model(data, model_save_name, isPair=False):

    if isPair:
        train_model.train_classification_model_with_pair(data, model_save_name)
    else:
        train_model.train_classification_model(data, model_save_name)


def predict_classification_model(train_data,
                                 test_data,
                                 model_save_name,
                                 isPair=False):

    if isPair:
        res = predict_model.predict_classification_model_with_pair(
            train_data, test_data, model_save_name)
    else:
        res = predict_model.predict_classification_model(
            train_data, test_data, model_save_name)
    return res


def find_noisy_label_drop(data,
                          model_save_name,
                          isPair=False,
                          times=5,
                          isDrop_noisy=True):

    final_noisy_list = []
    output = []

    if isPair:

        for i in range(times):
            output, noisy_list, data = find_noisy_labels.find_classification_noisy_labels_with_pair(
                data, model_save_name, isDrop_noisy=isDrop_noisy)
            final_noisy_list.extend(noisy_list)
    else:

        for i in range(times):
            output, noisy_list, data = find_noisy_labels.find_classification_noisy_labels(
                data, model_save_name, isDrop_noisy=isDrop_noisy)
            final_noisy_list.extend(noisy_list)

    noisy_dict = {}
    for nl in final_noisy_list:
        if nl not in noisy_dict:
            noisy_dict[nl] = 0
        noisy_dict[nl] += 1

    for d in data:
        noisy_id = d["noisy_id"]
        if noisy_id in noisy_dict and noisy_dict[noisy_id] > 1:
            output.append(d)

    fa.write_json("noisy_label.json", output, isIndent=True)

