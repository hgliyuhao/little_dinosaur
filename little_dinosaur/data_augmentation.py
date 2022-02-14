from itertools import count
import random
import fairies as fa
import jieba.analyse
import jieba
import copy 
import synonyms
import time
from tqdm import tqdm
from little_dinosaur import translate

def jieba_cut(text):
    
    """
        jieba分词普通模式
        传入text 返回list
    """
    res = []
    seg_list = jieba.cut(text, cut_all=False)
    for i in seg_list:
        res.append(i)

    return res  

def label_imbalance_augmentation(
    datasets,
    isBackTranslation = True,
    isReplaceSynonym = True,
    synonym_words_count = 1000
):
    """对多分类任务中，数量较少的类别进行数据增强

    通过同义词替换和回译的方式对数量较少的类别进行数据增强
    我们通过TF-IDF找到数量较少的类别中比较重要的词，通过同义词工具替换。
    与《Unsupervised Data Augmentation for Consistency Training》中的方法不同，在中文nlp任务中
    如果随机替换一些字词大概率会伤害到模型，哪怕替换的是通过TF-IDF找到的不重要的字。

    Parameters
    ----------
    datasets : 原始数据集
        需要确定数据集的格式,需要保证每条数据中包含"id","label"和"sentence"

        datasets = [
            {"id": 2214, "label": "2", "sentence": "我退货申请撤销 一下吧"},
            {"id": 1850, "label": "7", "sentence": "好的  谢谢，希望进完发货"}
        ]

    isBackTranslation : 默认为True
        是否使用回译的方式

    isReplaceSynonym : 默认为True
        是否使用关键词替换的方式

    synonym_words_count : int 默认为1000
        通过TF-IDF找到数量较少的类别中比较重要的词，通过同义词工具替换。选择需要替换的词语数量

    Returns
    ------- 
    和datasets格式一样的经过增强后的数据集

    """

    sentences,labels,ids = [],[],[]

    for d in datasets:
        sentences.append(d['sentence'])
        labels.append(d['label'])
        ids.append(int(d['id']))
    s = ','.join(sentences)
    tf_idf_all_words = jieba.analyse.extract_tags(s, topK = 50000, withWeight = False, allowPOS =())
    tf_idf_uninformative_words = tf_idf_all_words[-synonym_words_count:]

    counts = fa.count_label(labels)
    max_counts = max(counts.values())
    need_to_aug_label = []
    few_label = []
    for c in counts:
        if counts[c] < int(max_counts/6):
            few_label.append(c)
        elif counts[c] < int(max_counts/2):
            need_to_aug_label.append(c)

    add_data = []
    num_count = max(ids) + 100

    if isReplaceSynonym:
        for d in tqdm(datasets):
            temp = d
            label = d['label']
            sentence = d['sentence']
            if label in need_to_aug_label or label in few_label:
                new_words = copy.copy(tf_idf_uninformative_words)
                random.shuffle(new_words)
                seg_word_lists = jieba_cut(sentence)

                is_replace = False
                for word in seg_word_lists:
                    if word in new_words:
                        o = synonyms.nearby(word, 10)

                        if len(o[0]) > 0:
                            for synonyms_word in o[0]:
                                if synonyms_word != word:
                                    new_sentence = sentence.replace(word,synonyms_word)
                                    is_replace = True
                                    num_count += 1
                                    temp['sentence'] = new_sentence
                                    temp["id"] = num_count
                                    add_data.append(temp)
                                    break

                    if is_replace:
                        break

    if isBackTranslation:
        need_to_translate = {}
        translate_lists = []
        translate_back = {}  # 回译字典
        translate_back_to_zh = {}  # 回译字典英汉对照字典
        translate_back_lists = []

        for d in tqdm(datasets):
            temp = d
            label = d['label']
            sentence = d['sentence']
            id = d['id']
            if label in few_label:
                need_to_translate[sentence] = ''

        for sentence in need_to_translate:
            translate_lists.append(sentence)

        temp_translate = []
        for t in translate_lists:
            temp_translate.append(t)
            if len(temp_translate) == 200 :
                tran_res = translate.translate(temp_translate,'zh','en')
                for res in tran_res:
                    if res in need_to_translate:
                        need_to_translate[res] = tran_res[res]
                time.sleep(1)
                temp_translate = []
            elif t == translate_lists[-1]:
                tran_res = translate.translate(temp_translate,'zh','en')
                for res in tran_res:
                    if res in need_to_translate:
                        need_to_translate[res] = tran_res[res]
                time.sleep(1)
                temp_translate = []

        # fa.write_json('need_to.json',need_to_translate)
        for need in need_to_translate:
            translate_back_lists.append(need_to_translate[need])

        temp_translate = []
        for t in translate_back_lists:
            temp_translate.append(t)
            if len(temp_translate) == 200 :
                tran_res = translate.translate(temp_translate,'en','zh')
                for res in tran_res:
                    translate_back[res] = tran_res[res]
                time.sleep(1)
                temp_translate = []
            elif t == translate_lists[-1]:
                tran_res = translate.translate(temp_translate,'en','zh')
                for res in tran_res:
                    translate_back[res] = tran_res[res]
                time.sleep(1)
                temp_translate = []

        for res in need_to_translate:
            if need_to_translate[res] in translate_back:
                translate_back_to_zh[res] = translate_back[need_to_translate[res]]

        for d in tqdm(datasets):
            temp = d
            label = d['label']
            sentence = d['sentence']
            id = d['id']
            if label in few_label:
                if sentence in translate_back_to_zh and translate_back_to_zh[sentence] != '' and translate_back_to_zh[sentence] != sentence:
                    new_sentence = translate_back_to_zh[sentence]
                    num_count += 1
                    temp['sentence'] = new_sentence
                    temp["id"] = num_count
                    add_data.append(temp)

    return add_data        

