import pymysql

db = pymysql.connect(host='localhost', port=3306, user=username,passwd=pwd,db='2022_BBIBBO_DB', charset='utf8')
cursor = db.cursor()

sql = "INSERT INTO test (name ,email) VALUES('SION','sionzz1713@gmail.com')"
cursor.execute(sql)
db.commit()
db.close()

# CREATE TABLE TEST (
#             id INT UNSIGNED NOT NULL AUTO_INCREMENT,
#             name VARCHAR(20) NOT NULL,
#             email VARCHAR(20) NOT NULL,
#             PRIMARY KEY( id )
#         )