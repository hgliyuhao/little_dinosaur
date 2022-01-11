import fairies as fa
import random
import os
import shutil
import zipfile
from os.path import join, getsize

def random_split_data(fileName,test_size = 0.2):
    
    # 数据格式 "id": 2214, "label": "2", "sentence": "我退货申请撤销 一下吧"
    
    data = fa.read(fileName)
    labels,training_dict = [],{}
    for i in data:
        labels.append(i["label"])
        if i["label"] not in training_dict:
            training_dict[i["label"]] = []
        training_dict[i["label"]].append(i)  

    count_res = fa.count_label(labels)
    
    train_data,test_data = [],[]

    for c in count_res:
        new = training_dict[c]
        random.shuffle(new)
        split_position = int(count_res[c] * test_size)
        test_data.extend(new[:split_position])
        train_data.extend(new[split_position:])

    random.shuffle(test_data)
    random.shuffle(train_data)

    return train_data,test_data

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:     
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)       
    else:
        print('This is not zip')
# train_data,test_data = random_split_data("all.json")

# fa.write_json("train_data_fairies_1.json",train_data)
# fa.write_json("test_data_fairies_1.json",test_data)
