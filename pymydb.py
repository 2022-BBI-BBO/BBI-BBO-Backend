from ctypes.wintypes import BOOLEAN
from hashlib import new
from logging.handlers import DEFAULT_TCP_LOGGING_PORT
from pickle import TRUE
from unittest.mock import DEFAULT
import pymysql
def insert_db( path , smoke ): # path & smoke 
    db = pymysql.connect(host='localhost', port=3306, user=username,passwd=password,db='2022_BBIBBO_DB', charset='utf8')
    cursor = db.cursor()
    sql = f"INSERT INTO 2022_BBIBBO ( path , smoke ) VALUES('{path}' , {smoke} )" # val2 is boolean
    cursor.execute(sql)
    db.commit()
    db.close()

def select_db( column , tb_name ):
    result = []
    db = pymysql.connect(host='localhost', port=3306, user=username,passwd=password,db='2022_BBIBBO_DB', charset='utf8')
    cursor = db.cursor()
    sql = f"SELECT {column} FROM {tb_name}"
    cursor.execute(sql)
    res = cursor.fetchall() # recive 
    for data in res:
        data = ''.join(map(str,data)) # Converting an element into a string
        result.append(data) # data append
    db.commit()
    db.close()
    return result # return list