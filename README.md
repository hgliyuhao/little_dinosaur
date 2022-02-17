# little_dinosaur

用于中文NLP任务，数据增强，数据为中心处理工具  

* 针对中文语料做数据增强
- 针对不同的NLP任务，做有针对性的数据增强
* 利用算法自动检测出数据集中的错误标签

# 需要增强
目前针对噪音较大的数据集效果较好，但是需要样本均衡，如果样本不均衡，对于相近的两个样本结果会非常不友好

# 安装  
pip install little_dinosaur


# 参考论文  
Confident Learning: Estimating Uncertainty in Dataset Labels  
https://arxiv.org/abs/1911.00068  
Improving Generalization by Controlling Label-Noise Information in Neural Network Weights  
https://arxiv.org/abs/2002.07933   
Unsupervised Data Augmentation for Consistency Training  
https://arxiv.org/pdf/1904.12848v2.pdf  

# 环境

tensorflow <= 2.1.0 
keras == 2.3.1

<!-- # 主要功能
* txt,json,excel处理函数
- pdf抽取接口
* nlp常用工具 -->
# 数据格式

为保证工具正常工作，请参考example.json 需要保证每条数据都是一个字典 且字典中包含"id","label"和"sentence"  

{"id": 2214, "label": "2", "sentence": "我退货申请撤销 一下吧"}    
{"id": 1850, "label": "7", "sentence": "好的  谢谢，希望进完发货"}  

# 使用  
目前只适用文本分类任务

```python

# 你可以使用下面代码快速寻找错误标签

import little_dinosaur as ld

ld.eat(
        'example.json', # 你的数据集路径
        true_rate = 0.3,
        k_flod_times = 1,
        test_size = 0.1
    )

# 或者    
ld.self_learning(
    'example.json', # 你的数据集路径
    other_pre_model = False,
    maxlen = 48,
    batch_size = 96,
    epochs = 20,
    isPair = False,
    model_name = 'bert',
    test_size = 0.5,
    model_path = 'model_self_learning/',
    learning_rate = 3e-4,
    config_path = "",
    checkpoint_path = "",
    dict_path = "",
)


#  你可以使用下面的方式快速微调

ld.train_classification_model(

        fileName,
        other_pre_model = False,
        maxlen = 48,
        batch_size = 96,
        epochs = 40,
        isPair = False,
        model_name = 'bert',
        test_size = 0.2,
        model_path = 'model/',
        learning_rate = 2e-4

    )

#  也可以加载其他预训练模型微调

p = '/home/pre_models/chinese-roberta-wwm-ext-tf/'
config_path = p +'bert_config.json'
checkpoint_path = p + 'bert_model.ckpt'
dict_path = p +'vocab.txt'

ld.train_classification_model(
        fileName,
        other_pre_model = True,
        maxlen = 48,
        batch_size = 96,
        epochs = 40,
        isPair = False,
        model_name = 'bert',
        test_size = 0.2,
        model_path = 'model/',
        learning_rate = 2e-4,
        config_path = config_path,
        checkpoint_path = checkpoint_path,
        dict_path = dict_path,
    )

```
# TODO
数据集格式说明  
参数说明  
更新PU Learning 和PN Learning  
效果展示  
对单条数据进行数据增强  

# 更新

2022/2/14 新增数据增强方法 主要针对多分类任务中的稀疏样本  
2022/1/14 新增self_learning 更快的寻找错误标签的方式  
2022/1/12 修改内置预训练模型的下载方式  
2022/1/7 通过置信学习实现错误标签检测  


