# from little_dinosaur.download_utils import *
from download_utils import *
from load_pre_model import *
from confident_learning import *
import fairies as fa

def eat(
        pre_training_path,
        fileName
    ):
    
    if not download_verify(pre_training_path):
        download_pretrain_model(pre_training_path)

    # ToDO 

    output = find_noisy_label_by_confident_learning(fileName,pre_training_path,k_flod_times = 1,
        k_flod_num = 8)
    fa.write_npy('output.npy',output)

    # 现在是每个模型学习的结果，后面就是主动学习的处理
    # 写到log中


eat("test",r"D:\Ai\数据中心\upload\all.json")

