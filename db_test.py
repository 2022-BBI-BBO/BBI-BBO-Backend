from hashlib import new
import pymysql

db = pymysql.connect(host='localhost', port=3306, user=username,passwd=password,db='2022_BBIBBO_DB', charset='utf8')
cursor = db.cursor()
new_res = []
sql = "SELECT * FROM TEST"
cursor.execute(sql)
res = cursor.fetchall()
print(f'res type is {type(res)}')
print(f'res element tpye is {type(res[0])}')
for data in res:
    print(f'first data {data}')
    data = ''.join(map(str,data))
    print(f'second data {data}')
    # print(type(data))
    new_res.append(data)
print(f'This is res {res}')
print(new_res)
for data in new_res:
    print(data)
db.commit()
db.close()


# INSERT INTO test (name ,email) VALUES('SION','sionzz1713@gmail.com')

# CREATE TABLE TEST (
#             id INT UNSIGNED NOT NULL AUTO_INCREMENT,
#             name VARCHAR(20) NOT NULL,
#             email VARCHAR(20) NOT NULL,
#             PRIMARY KEY( id )
#         )