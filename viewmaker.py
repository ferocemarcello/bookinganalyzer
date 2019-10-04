from db import db_connection, db_operator

class viemaker:
    @staticmethod
    def do(originfile):
        f = open(originfile, "r")
        for line in f:
            liketextgood= ''
            liketextbad = ''
            kw=line[:-1]  # important to have last line with \n
            fs = open("subkeywords_booking/subkeywords_booking_cleaned/"+kw+".txt", "r")
            l=[]
            for subkw in fs:
                l.append(subkw)
                if '\'' in subkw:
                    subkw=subkw.replace('\'',"''")
                subkw=subkw[:-1]
                if subkw!='':
                    liketextgood+= ' or good LIKE \'%' + subkw + '%\''
                    liketextbad += ' or bad LIKE \'%' + subkw + '%\''
            fs.close()
            liketextgood= liketextgood[4:]
            liketextgood+= ';'
            liketextbad = liketextbad[4:]
            liketextbad += ';'
            db = db_connection()
            queryexecutor = db_operator(db)
            db.connect()
            query_good = 'CREATE VIEW ' + kw + '_good_view AS SELECT * FROM masterthesis.reviews where ' + liketextgood
            query_bad = 'CREATE VIEW ' + kw + '_bad_view AS SELECT * FROM masterthesis.reviews where ' + liketextbad
            print(query_bad)
            print(query_good)
            viewgood = queryexecutor.execute(query=query_good)
            viewbad = queryexecutor.execute(query=query_bad)
            db.disconnect()
        f.close()