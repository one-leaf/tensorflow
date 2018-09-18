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
    # time.sleep(0.5) # 抓取太快会被封IP   
    return r.text


curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, "data")
if not os.path.exists(data_dir): os.mkdir(data_dir)
fenlei = ["先秦","秦朝","汉朝","魏晋","南北朝","隋朝","唐朝","五代","宋朝","辽","金","元朝","明朝","清朝","近代"]
for f in fenlei:
    if f in ["先秦","秦朝"]: continue
    ffile = os.path.join(data_dir,"%s.txt"%f)   
    if os.path.exists(ffile): 
        continue
    poetry=[]    
    page = 1
    while True:
        params ={"page":page,"id":fenlei.index(f)+1,"dynasty":f}
        url = "http://www.4qx.net/Poetry_Dynasty.php"
        html = openurl(url,params)
        match_poetry_list = re.findall(r'<div class="main_text_2">(.*?)</div>',html,re.DOTALL)
        for poetry_html in match_poetry_list:
            poetry_lines_list = re.findall(r"<p>(.*?)</p>",poetry_html,re.DOTALL) 
            poetry_text=""
            for line in poetry_lines_list:
                if line.find(u"作者")>=0 or line.find(u"分类")>=0 or line.find("href")>=0:
                    continue
                if line.find("(")>=0:
                    line=re.sub(r'\(.*?\)','',line)
                if line.find("[")>=0:
                    line=re.sub(r'\[.*?\]','',line)                    
                poetry_text += line    
            poetry.append(poetry_text)  
        page += 1
        if html.find(u"末页")==-1:
            break    
    poetryfile = codecs.open(ffile,encoding='utf-8',mode='w')
    poetryfile.write(u"\r".join(poetry))
    poetryfile.close()

