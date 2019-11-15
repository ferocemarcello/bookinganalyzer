import time
import nltk
import spacy
import os,sys
import subprocess
from contextlib import contextmanager
from pycorenlp import StanfordCoreNLP
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
os.system('kill $(lsof -t -i:9000)')
os.chdir('./resources/stanford-corenlp-full-2018-10-05')
cmd = 'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 26000 &'
with open(os.devnull, "w") as f:
    subprocess.call(cmd, shell=True,stderr=f,stdout=f)
    os.chdir('../../')
    nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
    start_time = time.time()
    doc = "The chocolates great!"
    annot_doc = nlp_wrapper.annotate(doc,
        properties={
            'annotators': 'lemma, pos',
            'outputFormat': 'json',
            'timeout': 100000,
        })
f.close()
annot_doc=annot_doc['sentences'][0]['tokens']
print(str(time.time() - start_time))
os.system('kill $(lsof -t -i:9000)')
print("cacca")
'''nlp = spacy.load("en_core_web_sm")
f = open("resources/bow/allfreq/breakfast_good.txt", "r")
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
        print(i)
        thres+=0.1
f.close()
print(len(alltok))
print(len(notnounspacy))
print(len(notnounnltk))
print("cacca")'''
