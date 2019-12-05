import db
conn=db.db_connection()
conn.connect()
dbo=db.db_operator(conn)
good=[a[0] for a in dbo.execute('select Good from reviews limit 1000;')]
bad=[a[0] for a in dbo.execute('select Bad from reviews limit 1000;')]
conn.disconnect()
file1 = open("/home/marcelloferoce/Scrivania/goodthousand.txt","w")
for a in good:
    file1.write(a.replace('\n','')+'\n')
file1.close()
file2= open("/home/marcelloferoce/Scrivania/badthousand.txt","w")
for a in bad:
    file2.write(a.replace('\n','')+'\n')
file2.close()