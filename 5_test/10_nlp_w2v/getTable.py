#!/usr/bin/python3
'''
1. 从数据库下载报关单商品规范申报值和参数
'''

import psycopg2
import json
import os
import datetime

def save(conn, sql, sql_param, f):
    cur = conn.cursor()
    cur.execute(sql, sql_param)
    for r in cur:     
        sp=[r[0],r[1],r[2]]
        line = json.dumps(sp, ensure_ascii=False)
        f.write(line+"\n")
    
def main():
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "data")

    conn = psycopg2.connect("host=xxx.xxx.xxx.xxx port=5432 dbname=xxx user=xxx password=xxx")

    cur = conn.cursor()
    for table in ["decitemedi", "decitemhis"]:
        sql='''select min(createDate), max(createDate) from %s where declarefactor is not null 
            and hsmodel is not null and declarefactor<>'' and hsmodel<>''  ''' % table
        cur.execute(sql)
        min_date, max_date = cur.fetchone()
        min_date = min_date.date()
        max_date = max_date.date()
        print(min_date, max_date)

        # 删除文件
        file_name = os.path.join(data_dir,"%s.txt"%table)
        if os.path.exists(file_name): os.remove(file_name)
        f = open(file_name,'w', encoding="UTF-8")

        for day in range((max_date-min_date).days+1):
            curr_date = min_date+datetime.timedelta(day)
            print(curr_date)
            sql="select declarefactor, hsmodel, id from " + table + \
                " where declarefactor is not null and hsmodel is not null" + \
                " and declarefactor<> '' and hsmodel<>'' and createDate>%s and createDate<=%s"            
            save(conn, sql, (curr_date, curr_date + datetime.timedelta(1)), f)
            f.flush()
        f.close()
    conn.commit()
    conn.close()


if __name__ == '__main__':
    main()





