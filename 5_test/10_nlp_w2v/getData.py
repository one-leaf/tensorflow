#!/usr/bin/python3
'''
2. 将数据库下载报关单商品规范申报值和参数清洗，规范为DICT保存
'''

import psycopg2
import json
import os
import datetime
from utils import clearData

def save(srcfile, savefile):
    with open(srcfile,encoding="UTF-8") as f:
        for count, line in enumerate(f): 
            if count%10000==0: print(count, srcfile)  

            row = json.loads(line)
            declare_list = row[0].split(";")   #规范申报的项目
            hsmodel_list = row[1].split("|")   #企业的值
            hsmodel_split='|'
            id = row[2]

            # 如果企业的实际数据按|无法分割，尝试按;分割
            if len(hsmodel_list)==1:
                hsmodel_list = row[1].split(";")
                hsmodel_split=';'

            # 如果企业的实际数据按;无法分割，尝试按/分割
            if len(hsmodel_list)==1:
                hsmodel_list = row[1].split("/")
                hsmodel_split='/'

            # 如果分割后的企业数据的个数小于3个，则退出这一条, 太多为个了
            if len(hsmodel_list)<=2:
                # print(row)
                continue

            # 如果分割后只有3个并且不是用标准分割的，放弃掉
            if len(hsmodel_list)==3 and  hsmodel_split!='|':
                # print(row)
                continue

            declare_list, hsmodel_list = clearData(declare_list, hsmodel_list)
            if len(declare_list)==0 or len(hsmodel_list)==0: 
                # print(row)
                continue

            sp={}

            # 如果企业申报的项目每个都包含冒号，则按特殊处理
            maohao_count=0
            for hsmodel in hsmodel_list:
                if hsmodel.find(":")>0 or hsmodel.find("：")>0 or hsmodel=='':
                    maohao_count+=1

            # 如果企业数据按:分割的，直接采用企业的数据分割，不用管申报分类
            if len(hsmodel_list) - maohao_count<=1 and maohao_count>=3:
                for hsmodel in hsmodel_list:
                    key = ""
                    value=""
                    if hsmodel.find(":")>0:
                        key=hsmodel[:hsmodel.find(":")].strip()
                        value=hsmodel[hsmodel.find(":")+1:].strip()
                    elif hsmodel.find("：")>0:
                        key=hsmodel[:hsmodel.find("：")].strip()
                        value=hsmodel[hsmodel.find("：")+1:].strip()
                    if value in ["", "无", "无无", "无规格型号"]: continue
                    if key in ["", "无", ]: continue
                    sp[key]=value
            else:
                # 如果用户数据大于海关分类，此数据不可信，直接忽略
                # if len(hsmodel_list) > len(declare_list): 
                #     print(row)                
                #     continue
                off=0
                if "RHYTHM、CITIZEN等" in hsmodel_list and "品牌" in declare_list:
                    off = declare_list.index("品牌") - hsmodel_list.index("RHYTHM、CITIZEN等")
                if "无牌" in hsmodel_list and "品牌" in declare_list:
                    off = declare_list.index("品牌") - hsmodel_list.index("无牌")
                if "无牌子" in hsmodel_list and "品牌" in declare_list:
                    off = declare_list.index("品牌") - hsmodel_list.index("无牌子")
                if "无品牌" in hsmodel_list and "品牌" in declare_list:
                    off = declare_list.index("品牌") - hsmodel_list.index("无品牌")
                if "无型号" in hsmodel_list and "型号" in declare_list:
                    off = declare_list.index("型号") - hsmodel_list.index("无型号")

                for i, hsmodel in enumerate(hsmodel_list):  # 按企业数据循环
                    if hsmodel in ["", "无", "无无", "无规格型号"]: continue        # 如果用户输入的为空继续
                    d_index = i+off
                    if d_index >= len(declare_list) or d_index<0: continue       # 如果用户录入的超过了分类，中断
                    if declare_list[d_index]=="": continue         # 如果分类为空继续
                    # 如果栏位为其他，并且值用：分割，则采用里面的数据
                    if declare_list[d_index]=="其他" and hsmodel.find(":")>1:
                        sp[hsmodel[:hsmodel.find(":")]]=hsmodel[hsmodel.find(":")+1:]
                    else:
                        sp[declare_list[d_index]]=hsmodel

            # 如果数据只有1项对训练没啥好处，忽略； 好多数据都只有1项，不能忽略
            if len(sp)==0: 
                #  print(row)                
                 continue         
                                        
            sp['id']=id
            line = json.dumps(sp, ensure_ascii=False)
            savefile.write(line+"\n")        

def main():
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "data")

    file_name = os.path.join(data_dir,"decitem.txt")
    if os.path.exists(file_name): os.remove(file_name)
    with open(file_name, 'w', encoding="UTF-8") as f:
        for dfile in ["decitemedi.txt", "decitemhis.txt"]: 
            decitedmfile = os.path.join(data_dir, dfile)
            save(decitedmfile, f)
            f.flush()

if __name__ == '__main__':
    main()





