# little_dinosaur
利用算法自动检测出中文数据集中的错误标签

# 安装
pip install little_dinosaur

# 算法
Confident Learning: Estimating Uncertainty in Dataset Labels https://arxiv.org/abs/1911.00068  
Improving Generalization by Controlling Label-Noise Information in Neural Network Weights Improving Generalization by Controlling Label-Noise Information in Neural Network Weights  

<!-- # 主要功能
* txt,json,excel处理函数
- pdf抽取接口
* nlp常用工具 -->

代码示例

```python
# encoding=utf-8
import little_dinosaur as ld

    ld.eat(
        'all.json',
        'pre_training_path',
        true_rate = 0.3,
        k_flod_times = 1,
        test_size = 0.1
    )
```

# 更新

2022/1/7 通过置信学习实现错误标签检测  


