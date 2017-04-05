# coding=utf8
'''
抓取秋香网诗词存放
'''
import requests
import json
import re
import os
import time
import codecs

def openurl(url,params):
    print("GET",url,params)
    headers = {'User-agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.110 Safari/537.36',
               'Referer':'http://www.4qx.net/Poetry_Index.php/'}
    r = requests.get(url, headers=headers, params=params) 
    r.encoding = 'utf-8'   
    time.sleep(3) # 抓取太快会被封IP   
    return r.text


curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, "data")
if not os.path.exists(data_dir): os.mkdir(data_dir)
fenlei = ["先秦","秦朝","汉朝","魏晋","南北朝","隋朝","唐朝","五代","宋朝","辽","金","元朝","明朝","清朝","近代"]
for f in fenlei:
    ffile = os.path.join(data_dir,"%s.txt"%f)   
    if os.path.exists(ffile): 
        continue
    shige=[]    
    page = 1
    while True:
        params ={"page":page,"id":fenlei.index(f)+1,"dynasty":f}
        url = "http://www.4qx.net/Poetry_Dynasty.php"
        html = openurl(url,params)
        match_shige_list = re.findall(r'<div class="main_text_2">(.*?)</div>',html,re.DOTALL)
        for shige_html in match_shige_list:
            shige_lines_list = re.findall(r"<p>(.*?)</p>",shige_html,re.DOTALL) 
            for line in shige_lines_list:
                if line.find(u"作者")>=0 or line.find(u"分类")>=0 or line.find("href")>=0:
                    continue
                shige.append(line)  
        page += 1
        if html.find(u"末页")==-1:
            break    
    shigefile = codecs.open(ffile,encoding='utf-8',mode='w')
    shigefile.write(u"\r".join(shige))
    shigefile.close()

