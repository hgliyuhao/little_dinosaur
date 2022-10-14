import fairies as fa
from model_utils import train_model
from model_utils import predict_model
from model_utils import data_utils
from model_utils import find_noisy_labels


def train_classification_model(train_data,
                               model_save_name,
                               test_data=[],
                               isPair=False):

    id2label, label2id = data_utils.get_category(train_data, test_data)

    train_model.train_classification_model(train_data,
                                           model_save_name,
                                           id2label,
                                           label2id,
                                           isPair=isPair)


def predict_classification_model(test_data,
                                 model_save_name,
                                 train_data=[],
                                 isPair=False):

    id2label, label2id = data_utils.get_category(test_data, train_data)

    res = predict_model.predict_classification_model(test_data,
                                                     model_save_name,
                                                     id2label,
                                                     label2id,
                                                     isPair=isPair)

    return res


def find_noisy_label_drop(data,
                          model_save_name,
                          isPair=False,
                          times=5,
                          isDrop_noisy=True):

    final_noisy_list = []
    output = []

    id2label, label2id = data_utils.get_category(data, [])
    class_num = len(id2label)

    for i in range(times):
        output, noisy_list, data = find_noisy_labels.find_classification_noisy_labels(
            data,
            model_save_name,
            id2label,
            label2id,
            isPair=isPair,
            isDrop_noisy=isDrop_noisy)
        final_noisy_list.extend(noisy_list)

    noisy_dict = {}
    for nl in final_noisy_list:
        if nl not in noisy_dict:
            noisy_dict[nl] = 0
        noisy_dict[nl] += 1

    clean_data = []

    for d in data:

        noisy_id = d["noisy_id"]
        if noisy_id in noisy_dict and noisy_dict[noisy_id] > int(
                1 + times / class_num * 0.5):
            output.append(d)
        else:
            clean_data.append(d)

    # 用clean_data 重新训练
    id_dict = {}
    for i in range(5):
        train_classification_model(clean_data,
                                   model_save_name,
                                   test_data=data,
                                   isPair=isPair)
        res = predict_classification_model(data,
                                           model_save_name,
                                           train_data=data,
                                           isPair=isPair)

        for j in res:
            id = j["noisy_id"]
            if id not in id_dict:
                id_dict[id] = {}
            predict_label = j["score"]
            if predict_label not in id_dict[id]:
                id_dict[id][predict_label] = 0
            id_dict[id][predict_label] += 1

    retrained_noisy_label = []

    for d in data:
        id = d["noisy_id"]
        label = d["label"]
        predict_res = id_dict[id]
        sorted_predict_res = sorted(predict_res.items(), key=lambda x: x[1],reverse= True)
        for s in sorted_predict_res:
            if s[0] != label:
                d["trans_label"] = s[0]
                retrained_noisy_label.append(d)
            break

    fa.write_json("noisy_label.json", retrained_noisy_label, isIndent=True)
