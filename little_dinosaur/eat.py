# from little_dinosaur.download_utils import *
from download_utils import *
from load_pre_model import *
import config

def eat(
        pre_training_path,

    ):

    # download_pretrain_model(pre_training_path)
    print(load_pre_model(pre_training_path))

eat("test")

