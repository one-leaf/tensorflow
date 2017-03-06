# coding=utf8
'''
作废，没法抓取数据，会被禁止
'''
import urllib.request
import json
import re
import os

def openurl(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.110 Safari/537.36')]
    req = opener.open(url)
    handler = urllib.request.urlopen(req).read()
    return handler.read().decode("utf-8") 

# 获得电影列表
movies=[]
for page in range(10):
    url="https://movie.douban.com/j/search_subjects?type=movie&tag=%E7%83%AD%E9%97%A8&sort=rank&page_limit=20&page_start={}".format(page)
    html = openurl(url)
    j = json.loads(html)
    for x in j['subjects']:        
        movies.append((x["title"],float(x["rate"]),x["url"]))
    time.sleep(5)

out_dir = os.path.dirname(__file__)
data_dir = os.path.join(out_dir, "data")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# 保存电影数据
open(os.path.join(data_dir,"movies.json"),"w").write(json.dumps(movies))
starfiles=[]
for i in range(5):
    starfile = open(os.path.join(data_dir,"{}.txt".format((i+1)*10)),mode='w',encoding="utf8")
    starfiles.append(starfile)

#下载电源评论和打星表
for movie in movies:
    movie_name, movie_rate, movie_url =  movie
    moviefile = open(os.path.join(data_dir,"{}.txt".format(movie_name)),mode='w',encoding="utf8")
    html = urllib.request.urlopen(movie_url).read()
    html = html.decode("utf-8")
    match_votes = re.search(r'<span property="v:votes">(.*?)</span>',html)
    print(movie_name.encode("GB18030"))
    if match_votes:
        votes = int(match_votes.group(1))
        pages = votes//20
        for i in range(pages):
            print(movie_name.encode("GB18030"),pages,i)
            url=movie[2]+"collections?start={}".format(i*20)
            html = openurl(url)
            comments=re.findall(r'<p class="pl">.*?</td>',html,re.DOTALL)
            for comment in comments:
                match_c = re.search(r'(allstar..).*?<p>(.*?)</p>',comment,re.DOTALL)
                if match_c:
                    star,comm = match_c.groups()
                    comm=comm.strip()+"\n"
                    moviefile.write(comm)
                    if star.find("10")>0:
                        starfiles[0].write(comm)
                    elif star.find("20")>0:
                        starfiles[1].write(comm)
                    elif star.find("30")>0:
                        starfiles[2].write(comm)
                    elif star.find("40")>0:
                        starfiles[3].write(comm)
                    elif star.find("50")>0:
                        starfiles[4].write(comm)
            #抓取的太快会给禁止？                        
            time.sleep(5)
    moviefile.close()

for starfile in starfiles:
    starfile.close()
    # <span property="v:votes">19936</span>人评价</a>

