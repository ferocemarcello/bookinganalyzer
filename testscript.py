import time
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")
f = open("resources/bow/allfreq/stanford/bedroom_good.txt", "r")
tot=0
notnounspacy=[]
notnounnltk=[]
alltok=[]
i=0
thres=0.4
for r in f.readlines():
    i+=1
    tok=r.split(',')[1][2:-1]
    alltok.append(tok)
    posnltk=nltk.pos_tag([tok])
    posspacy=nlp(tok)[0].pos_
    if posnltk[0][1]!='NN':
        notnounnltk.append(r+'='+posnltk[0][1])
    if posspacy!='NOUN':
        notnounspacy.append(r+'='+posspacy)
    freq=float(r.split(',')[3])
    tot+=freq
    if tot>=thres:
        print(str(tot)+' first '+str(i)+' elements')
        thres+=0.1
f.close()
print(len(alltok))
print(len(notnounspacy))
print(len(notnounnltk))
print("cacca")
