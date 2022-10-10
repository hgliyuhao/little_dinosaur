import fairies as fa
from tqdm import tqdm
import configparser


def read_config(conf_path):
    """
    根据配置文件的路径读取配置文件，并返回配置文件内容组成的dict
    :param conf_path: 配置文件路径
    :return: 配置文件组成的dict
    """
    conf_dict = {}
    cf = configparser.ConfigParser()
    cf.read(conf_path, encoding='utf-8')
    secs = cf.sections()
    for s in secs:
        items = cf.items(s)
        for i in items:
            conf_dict[i[0]] = i[1]
    return conf_dict


def convertNumToChinese(text):
    uppercase_numbers = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i in text:
        if i in nums:
            text = text.replace(i, uppercase_numbers[nums.index(i)])
    return text


def clean_text(text):

    text = fa.cht_2_chs(str(text))
    text = fa.strQ2B(str(text))
    text = text.lower()
    anti_sign = [' ', ' ', '.', '.', '．', '·', '-', '——', '－']
    text = text.replace("（", '(').replace('）', ')')
    for anti in anti_sign:
        text = text.replace(anti, '')

    text = convertNumToChinese(text)

    # todo
    # （）转成()
    # 数字转成文字
    # 处理.的问题
    # 去掉空格

    return text


def containEnglish(str0):
    import re
    return bool(re.search('[a-z]', str0))
