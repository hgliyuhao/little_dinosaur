from little_dinosaur import utils

import git
from git import RemoteProgress
from tqdm import tqdm
import os
import requests

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

def DownloadFile(save_url,file_name):
        
        url = "http://139.162.18.124/data/pre_training_model.zip"
        # 文件夹不存在，则创建文件夹
        folder = os.path.exists(save_url)
        if not folder:
            os.makedirs(save_url)
        res = requests.get(url,stream=True) 
        total_size = int(int(res.headers["Content-Length"])/1024+0.5)
        # 获取文件地址
        file_path = os.path.join(save_url, file_name)
        
        # 打开本地文件夹路径file_path，以二进制流方式写入，保存到本地
        from tqdm import tqdm
        with open(file_path, 'wb') as fd:
            print('开始下载文件：{},当前文件大小：{}KB'.format(file_name,total_size))
            for chunk in tqdm(iterable=res.iter_content(1024),total=total_size,unit='k',desc=None):
                fd.write(chunk)
            print(file_name+' 下载完成！')

        dst_dir = os.path.join(save_url, 'pre_model')
        utils.unzip_file(file_path, dst_dir)
        if os.path.exists(file_path):
            os.remove(file_path)

def download_pretrain_model(pre_training_path):
   
    # print("  ")
    # print("start downloading...")
    # print("  ")
    
    # git.Repo.clone_from('https://github.com/hgliyuhao/pre_training_model.git', pre_training_path, branch='main', progress=CloneProgress())
    try :
        DownloadFile(pre_training_path,'pre_model.zip')
    except:
        git.Repo.clone_from('https://github.com/hgliyuhao/pre_training_model.git', pre_training_path, branch='main', progress=CloneProgress())    



def download_verify(pre_training_path):
    
    dir_path = os.path.dirname(os.path.abspath(__file__))
    verify_list = [
        'pre_model/albert_tiny_zh_google/albert_config_tiny_g.json', 
        'pre_model/albert_tiny_zh_google/albert_model.ckpt.data-00000-of-00001', 
        'pre_model/albert_tiny_zh_google/albert_model.ckpt.index', 
        'pre_model/albert_tiny_zh_google/albert_model.ckpt.meta', 
        'pre_model/albert_tiny_zh_google/checkpoint', 
        'pre_model/albert_tiny_zh_google/vocab.txt', 
        # 'pre_model/electra-small/bert_config_tiny.json', 
        # 'pre_model/electra-small/checkpoint', 
        # 'pre_model/electra-small/electra_small.data-00000-of-00001', 
        # 'pre_model/electra-small/electra_small.index', 
        # 'pre_model/electra-small/vocab.txt', 
        'pre_model/electra_180g_small/electra_180g_small.ckpt.data-00000-of-00001', 
        'pre_model/electra_180g_small/electra_180g_small.ckpt.index', 
        'pre_model/electra_180g_small/electra_180g_small.ckpt.meta', 
        'pre_model/electra_180g_small/small_discriminator_config.json', 
        'pre_model/electra_180g_small/small_generator_config.json', 
        'pre_model/electra_180g_small/vocab.txt', 
        'pre_model/electra_180g_small/__MACOSX',
        # 'pre_model/rbtl3/bert_config_rbtl3.json',
        # 'pre_model/rbtl3/bert_model.ckpt.data-00000-of-00001',
        # 'pre_model/rbtl3/bert_model.ckpt.index',
        # 'pre_model/rbtl3/bert_model.ckpt.meta',
        # 'pre_model/rbtl3/vocab.txt'
    ]
    for path in verify_list:
        new = os.path.join(pre_training_path,path)
        dir_path_new = os.path.join(dir_path,path)
        if not os.path.exists(new) and not os.path.exists(dir_path_new):
            return False
    return True
