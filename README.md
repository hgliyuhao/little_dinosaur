# little_dinosaur
利用算法自动检测出中文数据集中的错误标签

# 安装  
pip install little_dinosaur

# 论文  
Confident Learning: Estimating Uncertainty in Dataset Labels  
https://arxiv.org/abs/1911.00068  
Improving Generalization by Controlling Label-Noise Information in Neural Network Weights  
https://arxiv.org/abs/2002.07933  


<!-- # 主要功能
* txt,json,excel处理函数
- pdf抽取接口
* nlp常用工具 -->
# 使用  
代码示例

```python

# 你可以使用下面代码快速寻找错误标签

import little_dinosaur as ld

ld.eat(
        'all.json', # 你的数据集路径
        true_rate = 0.3,
        k_flod_times = 1,
        test_size = 0.1
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
        test_size = 0.001,
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
        other_pre_model = False,
        maxlen = 48,
        batch_size = 96,
        epochs = 40,
        isPair = False,
        model_name = 'bert',
        test_size = 0.001,
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



# 更新

2022/1/12 修改内置预训练模型的下载方式
2022/1/7 通过置信学习实现错误标签检测  


