from pickle import encode_long
import pandas as pd
import nltk
from pprint import pprint
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.text import Text
import re
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as nump


def text_to_tokens(text_seqs):
    token_seqs = []
    for i in text_seqs:
      x = i.lower().split()
      token_seqs.append(x)
    return token_seqs

filtered_p=pd.read_csv(r'C:/Users/aniur/Desktop/NLP REVIEW 3/Review 3/cleaned_data_pos.csv')
filtered_n=pd.read_csv(r'C:/Users/aniur/Desktop/NLP REVIEW 3/Review 3/cleaned_data_neg.csv')
p1=filtered_p['0']
n1=filtered_n['0']

pt=text_to_tokens(p1)
nt=text_to_tokens(n1)
N=[]
P=[]
sia=SentimentIntensityAnalyzer()
for t in nt:
    for s in t:
        A=sia.polarity_scores(s)
        print(A)
        if(A['neg']==1.0):
            N.append(s)
for t in pt:
    for s in t:
        B=sia.polarity_scores(s)
        print(B)
        if(B['pos']==1.0):
            P.append(s)     

ncount=0
pcount=0
All_Neg = []
for i in N:
    if i not in All_Neg:
        All_Neg.append(i)
        ncount=ncount+1
All_Pos = []
for i in P:
    if i not in All_Pos:
        All_Pos.append(i)
        pcount=pcount+1

print(All_Neg)
print(All_Pos)
AN=pd.DataFrame(All_Neg)
AP=pd.DataFrame(All_Pos)

AN.to_csv(r'C:/Users/aniur/Desktop/NLP REVIEW 3/Review 3/trained_neg.csv')
AP.to_csv(r'C:/Users/aniur/Desktop/NLP REVIEW 3/Review 3/trained_pos.csv')

y=nump.array([ncount,pcount])
ml=["Negative","Positive"]
plt.pie(y, labels = ml,autopct='%1.2f%%')
plt.show()