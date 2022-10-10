import fairies as fa
from model_utils import *
from model_utils import train_model
from model_utils import predict_model

# def find_noisy_label
# data = fa.read("data/train_name_label_data.json")

# for i in range(10):

#     train_model.train_classification_model_with_pair(data,'model/temp.weights')
#     res = predict_model.predict_classification_model_with_pair(data, data, 'model/temp.weights')
#     fa.write_npy("res_{}.json".format(i),res)

# data = fa.read("data/train_name_label_data.json")

# data = data[:1000]

# res = train_model.train_classification_model_with_pair(data,
#                                                        'model/temp.weights',
#                                                        isDrop_noisy=True)

# fa.write_json("noisy_label.json", res, isIndent=True)

# test noisy_label


def find_noisy_label_drop():

    times = 5

    data = fa.read("data/train_name_label_data.json")

    final_noisy_list = []
    output = []

    for i in range(times):
        output, noisy_list, data = train_model.train_classification_model_with_pair(
            data, 'model/temp.weights', isDrop_noisy=True)
        final_noisy_list.extend(noisy_list)

    # final_noisy_list = list(set(final_noisy_list))
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

find_noisy_label_drop()