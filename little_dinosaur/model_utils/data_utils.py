import fairies as fa

# {"id": 2214, "label": "2", "sentence": "我退货申请撤销 一下吧"}
# {"id": 1850, "label": "7", "sentence": "好的 谢谢，希望进完发货"}

# {"id": 2214, "label": "2", "sentence_1": "我退货申请撤销 一下吧", "sentence_2": "好的 谢谢，希望进完发货"},


def convert_data_to_train(data):

    data = add_id(data)
    return data


def add_id(data):

    id_data = data.copy()

    count = 0
    for d in id_data:
        d["noisy_id"] = count
        count += 1

    return id_data


def get_category(train_data, test_data):

    data = train_data + test_data

    category = set()
    for d in data:
        category.add(d["label"])

    category = list(category)
    category.sort()
    id2label, label2id = fa.label2id(category)

    return id2label, label2id


def load_data(data, id2label, label2id, test_size=0.1, isPair=False):

    train_data, test_data = [], []

    train_lists, test_lists = fa.split_classification_data(data, test_size)

    if isPair:

        for d in train_lists:
            sentence_1 = d["sentence_1"]
            sentence_2 = d["sentence_2"]
            label = label2id[d["label"]]
            noisy_id = d["noisy_id"]
            train_data.append([sentence_1, sentence_2, label, noisy_id])

        for d in test_lists:
            sentence_1 = d["sentence_1"]
            sentence_2 = d["sentence_2"]
            label = label2id[d["label"]]
            noisy_id = d["noisy_id"]
            test_data.append([sentence_1, sentence_2, label, noisy_id])
        return train_data, test_data
    else:

        for d in train_lists:
            sentence = d["sentence"]
            label = label2id[d["label"]]
            noisy_id = d["noisy_id"]
            train_data.append([sentence, label, noisy_id])

        for d in test_lists:
            sentence = d["sentence"]
            label = label2id[d["label"]]
            noisy_id = d["noisy_id"]
            test_data.append([sentence, label, noisy_id])

        return train_data, test_data


def load_test_data(test_data, id2label, label2id, isPair=False):

    test = []

    if isPair:
        for d in test_data:
            sentence_1 = d["sentence_1"]
            sentence_2 = d["sentence_2"]
            label = label2id[d["label"]]
            noisy_id = d["noisy_id"]
            test.append([sentence_1, sentence_2, label, noisy_id])

    else:
        for d in test_data:
            sentence = d["sentence"]
            label = label2id[d["label"]]
            noisy_id = d["noisy_id"]
            test.append([sentence, label, noisy_id])

    return test
