import time
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")
f = open("resources/bow/allfreq/stanford/breakfast_good.txt", "r")
tot=0
notnounspacy=[]
notnounnltk=[]
alltok=[]
i=0
thres=0.4
slash=[]
dash=[]
notalpha=[]
dec=[]
exmark=[]
quemark=[]
freqsbag=0
for r in f.readlines():
    appended = False
    i+=1
    tok=r.split(',')[1][2:-1]
    if '\\' in tok or '/' in tok:
        slash.append(tok)
        freqsbag+=float(r.split(',')[3])
        appended=True
    if '-' in tok and not appended:
        dash.append(tok)
        appended = True
        freqsbag += float(r.split(',')[3])
    if '!' in tok and not appended:
        exmark.append(tok)
        freqsbag += float(r.split(',')[3])
        appended = True
    if '?' in tok and not appended:
        quemark.append(tok)
        appended = True
        freqsbag += float(r.split(',')[3])
    if not tok.isalnum() and not appended:
        notalpha.append(tok)
        freqsbag += float(r.split(',')[3])
        appended = True
    if tok.isdecimal() and not appended:
        dec.append(tok)
        freqsbag += float(r.split(',')[3])
        appended = True

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
print(freqsbag)
print(len(alltok))
print(len(notnounspacy))
print(len(notnounnltk))
print(len(slash))
print(len(dash))
print(len(exmark))
print(len(quemark))
print(len(notalpha))
print(len(dec))
print("cacca")
