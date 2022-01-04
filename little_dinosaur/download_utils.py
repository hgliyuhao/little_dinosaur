import git
from git import RemoteProgress
from tqdm import tqdm
import os


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
   
    print("  ")
    print("start downloading...")
    print("  ")
    
    git.Repo.clone_from('https://github.com/hgliyuhao/pre_training_model.git', pre_training_path, branch='main', progress=CloneProgress())

def download_verify(pre_training_path):
    
    dir_path = os.path.dirname(os.path.abspath(__file__))
    verify_list = ['pre_model\\albert_tiny_zh_google\\albert_config_tiny_g.json', 'pre_model\\albert_tiny_zh_google\\albert_model.ckpt.data-00000-of-00001', 
'pre_model\\albert_tiny_zh_google\\albert_model.ckpt.index', 'pre_model\\albert_tiny_zh_google\\albert_model.ckpt.meta', 'pre_model\\albert_tiny_zh_google\\checkpoint', 'pre_model\\albert_tiny_zh_google\\vocab.txt', 'pre_model\\electra-small\\bert_config_tiny.json', 'pre_model\\electra-small\\checkpoint', 'pre_model\\electra-small\\electra_small.data-00000-of-00001', 'pre_model\\electra-small\\electra_small.index', 'pre_model\\electra-small\\vocab.txt', 'pre_model\\electra_180g_small\\electra_180g_small.ckpt.data-00000-of-00001', 'pre_model\\electra_180g_small\\electra_180g_small.ckpt.index', 'pre_model\\electra_180g_small\\electra_180g_small.ckpt.meta', 'pre_model\\electra_180g_small\\small_discriminator_config.json', 'pre_model\\electra_180g_small\\small_generator_config.json', 'pre_model\\electra_180g_small\\vocab.txt', 'pre_model\\electra_180g_small\\__MACOSX']
    for path in verify_list:
        new = os.path.join(pre_training_path,path)
        dir_path_new = os.path.join(dir_path,path)
        if not os.path.exists(new) and not os.path.exists(dir_path_new):
            return False
    return True
