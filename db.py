import mysql.connector
from mysql.connector import ProgrammingError


class db_connection:
    def __init__(self,user='masteruser',password='mining',host='127.0.0.1',database='masterthesis',usepure=True):
        self.config={
        'user': user,
        'password': password,
        'host': host,
        'database': database,
        'raise_on_warnings': True,
        'use_pure': usepure,
        'port': 3306,
    }
    def connect(self):
        self.connection = mysql.connector.connect(**self.config)
    def disconnect(self):
        self.connection.close()
class db_operator:
    def __init__(self,db_con):
        self.dbconnection=db_con
    def execute(self, query):
        cursor = self.dbconnection.connection.cursor()
        try:
            cursor.execute(query)
        except ProgrammingError as e:
            print(e.msg)
            return None
        rows=[]
        for t in cursor:
            rows.append(t)
        cursor.close()
        return rows