import imp
import git
from git import RemoteProgress
from tqdm import tqdm

class CloneProgress(RemoteProgress):
    
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()
        self.op_code = -1
        self.max_count = -1
    def update(self, op_code, cur_count, max_count=None, message=''):
        if op_code != self.op_code and self.max_count != max_count:
            self.max_count = max_count
            self.op_code = op_code
            print("\n")
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()


def download_pretrain_model(pre_training_path):
   
    # TODO 去历史记录里面找

    print("start downloading...")
    print("  ")
    
    git.Repo.clone_from('https://github.com/hgliyuhao/pre_training_model.git', pre_training_path, branch='main', progress=CloneProgress())

import os
def get_listdir(path):
    
    """
        得到文件下路径
    """

    res = []
    a = os.listdir(path)
    for i in a:
        k = os.path.join(path,i)
        res.append(k)

    return res   

a = get_listdir('test/pre_model')    

for i in a:

    print(get_listdir(i))