# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
from hashlib import md5

def translate(query_lists, from_lang = 'en',to_lang =  'zh'):
    
    """
        how to use    
        a = ['yes','no']
        b = translate(a)


    """
    
    # Set your own appid/appkey.
    appid = '20201107000610595'
    appkey = 'EylFfm6wOcWQRNaI3LZ9'

    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`


    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    query = '\n'.join(query_lists)

    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    res = {}
    for dst in result["trans_result"]:
        res[dst["src"]] = dst["dst"]
    return res
