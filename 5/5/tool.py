# coding=utf-8

import os

curr_dir = os.path.dirname(__file__)

files=os.listdir(os.path.join(curr_dir,'midi','song'))

i=251
for d in files:
    dirname = os.path.join(curr_dir,'midi','song',d)
    if not dirname.endswith('MID'): continue
    for f in os.listdir(dirname):
        filename = os.path.join(dirname,f)
        if not filename.endswith('mid'): continue
        distname = os.path.join(curr_dir,'midi','0%s.MID'%i)
        os.rename(filename,distname)
        i=i+1 
        print(filename)