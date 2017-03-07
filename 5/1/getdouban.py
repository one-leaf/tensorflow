# coding=utf8
'''
抓取豆瓣前2000部电影的评论，并按打分情况存放
'''
import requests
import json
import re
import os
import time
import codecs

def openurl(url):
    print("GET",url)
    headers = {'User-agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.110 Safari/537.36',
               'Referer':'https://movie.douban.com/'}
    r = requests.get(url, headers=headers)    
    time.sleep(5) # 抓取太快会被封IP   
    return r.text

# 获得前2000个电影列表，如果之前已经下载过了，就直接加载
curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, "data")
if not os.path.exists(data_dir): os.mkdir(data_dir)
movies_file = os.path.join(data_dir,"movies.json")
if os.path.exists(movies_file):
    movies=json.loads(codecs.open(movies_file,encoding='utf-8').read())
else:
    movies=[]
    for page in range(100):
        url="https://movie.douban.com/j/search_subjects?type=movie&tag=%E7%83%AD%E9%97%A8&sort=rank&page_limit=20&page_start={}".format(page*20)
        html = openurl(url)
        j = json.loads(html)
        for x in j['subjects']:        
            movies.append((x["title"],float(x["rate"]),x["url"]))
    # 保存电影数据
    codecs.open(os.path.join(data_dir,"movies.json"),encoding='utf-8',mode="w").write(json.dumps(movies))

# 下载电源评论和打星表
for movie in movies:
    movie_name, movie_rate, movie_url =  movie
    moviefilename = os.path.join(data_dir,u"{}.txt".format(movie_name))
    if os.path.exists(moviefilename): continue
    html = openurl(movie_url)
    match_votes = re.search(r'<span property="v:votes">(.*?)</span>',html)
    vote_comment_list=[]
    if match_votes:
        votes = int(match_votes.group(1))
        pages = votes//20
        for i in range(pages):
            url=movie[2]+"collections?start={}".format(i*20)
            html = openurl(url)
            comments=re.findall(r'<p class="pl">.*?</td>',html,re.DOTALL)
            for comment in comments:
                match_c = re.search(r'(allstar..).*?<p>(.*?)</p>',comment,re.DOTALL)
                if match_c:
                    star,comm = match_c.groups()
                    vote_comment_list.append((star,comm))
            # 如果没有评论了，后面的也不看了
            if len(comments)<20: break 
    moviefile = codecs.open(moviefilename,encoding='utf-8',mode='w')
    moviefile.write(json.dumps(vote_comment_list))
    moviefile.close()

