# fairies  
utils for nlp

# 安装
pip install little_dinosaur

# 主要功能
* txt,json,excel处理函数
- pdf抽取接口
* nlp常用工具

# 常用API

**label2id**
用于序列标注时标签和id相互转换  
**find_lcs**
查找公共子串  
**random_build_data**
按照比例切分数据集  
**removeLineFeed**
清洗数据中的去除换行 tab键  
**text_len_analysis**
分析数据中的文本长度  
**split_to_paragraph**
将文本切成句子  
**get_slide_window_text**
滑动窗口切割句子  
**find_co_occurrence_word**
通过统计词语的共现次数,寻找关键词  
**chs_2_cht**  
简体到繁体  
**strQ2B**
全角转半角  
**long_substr**
多个字符的公共子串  

# 更新日志

2021/9/21 添加简体到繁体, 全角转半角,多个字符的公共子串,添加同音,同形,以及汉字字表
2021/6/2 添加find_co_occurrence_word方法,通过统计词语的共现次数,寻找关键词  
2021/6/2 添加get_cut_window_text方法,使用截断的方法切割句子,用于处理nlp工作中超长文本的处理  
2021/4/21 添加get_slide_window_text方法,使用滑动窗口切割句子,用于处理nlp工作中超长文本的处理    


