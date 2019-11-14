import nltk
import spacy

nlp = spacy.load("en_core_web_sm")
f = open("resources/bow/allfreq/breakfast_good.txt", "r")
tot=0
notnounspacy=[]
notnounnltk=[]
alltok=[]
for r in f.readlines():
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
f.close()
print(len(alltok))
print(len(notnounspacy))
print(len(notnounnltk))
print("cacca")
