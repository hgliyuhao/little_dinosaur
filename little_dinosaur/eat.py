# from little_dinosaur.download_utils import *
from download_utils import *
from load_pre_model import *
from confident_learning import *
import fairies as fa
import os

def eat(

        fileName,
        pre_training_path,
        true_rate = 0.3,
        k_flod_times = 10,
        test_size = 0.1
    
    ):
    
    if not download_verify(pre_training_path):
        download_pretrain_model(pre_training_path)

    output = find_noisy_label_by_confident_learning(
        fileName,
        pre_training_path,
        k_flod_times = k_flod_times,
        test_size = test_size
    )

    if not os.path.exists("log"):
        os.makedirs("log")

    fa.write_npy('log/output.npy',output)

    # 现在是每个模型学习的结果，后面就是主动学习的处理
    # 写到log中

    all = fa.read_json(fileName)

    id2label = {}

    for right in all:
        id2label[right['id']] = int(right['label'])

    res = {}

    for i in output:

        for j in i:
            if j not in res:
                res[j] = []
            for r in  i[j]:
                new = np.array(r)
                
                if new.max() > true_rate :
                    res[j].append(new.argmax(axis=0))

    maybe_wrong = []

    for j in res:
        if len(res[j]) > 1 and len(list(set(res[j]))) ==1:
            if id2label[j]  not in res[j]:
                maybe_wrong.append(j)

    fa.write_json('log/wrong_case.json',maybe_wrong)



# eat("test","all.json")

