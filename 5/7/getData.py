# coding=utf-8

import os
import urllib.request

url1='https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/Data/CAT_DATASET_01.zip'
url2='https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/Data/CAT_DATASET_02.zip'
url3='https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/Data/00000003_015.jpg.cat'
curr_dir = os.path.dirname(__file__)
file1 = os.path.join(curr_dir,'data','CAT_DATASET_01.zip')
file2 = os.path.join(curr_dir,'data','CAT_DATASET_02.zip')
file3 = os.path.join(curr_dir,'data','00000003_015.jpg.cat')


def download(url,savefile):
    raise("下载不下来，自己找迅雷下载", url, '保存为', savefile)
    urllib.request.urlretrieve(url, savefile)


def main():   
    if not os.path.exists(file1):
        print("start download",file1)
        download(url1,file1)
    if not os.path.exists(file2):
        print("start download",file2)
        download(url2,file2)
    if not os.path.exists(file3):
        print("start download",file3)
        download(url3,file3)    


if __name__ == '__main__':
    main()