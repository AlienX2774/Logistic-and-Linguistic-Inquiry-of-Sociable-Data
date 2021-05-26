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

sia=SentimentIntensityAnalyzer()
nonroot_file1=pd.read_csv(r'E:\VIT\Semester-4\CSE4022 (NLP)\Project\Codes\Dataset\scraped_data.csv')
p=[]
n=[]
score=0
nonroot_file = nonroot_file1['Text']

for sen in nonroot_file:
  score=sia.polarity_scores(sen)
  print(score , "\n")
  if (score['pos']<score['neg']):
    n.append(sen)
  else:
    p.append(sen)

p_file=pd.DataFrame(p)
p_file.to_csv(r'E:\VIT\Semester-4\CSE4022 (NLP)\Project\Codes\Dataset\positive_data.csv')
n_file=pd.DataFrame(n)
n_file.to_csv(r'E:\VIT\Semester-4\CSE4022 (NLP)\Project\Codes\Dataset\negative_data.csv')