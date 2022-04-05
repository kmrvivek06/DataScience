import pymysql


def connectdb():
     conn = pymysql.connect(host='127.0.0.1', port=3306, user='root',password='lenova14',db='lenova')
     return conn
    