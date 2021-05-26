from pickle import encode_long
import pandas as pd
import nltk
from pprint import pprint
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.text import Text
#import spacy
import re
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords
#encoder = spacy.load('en_core_web_sm')

nltk.download(["names","stopwords","state_union","twitter_samples","movie_reviews","averaged_perceptron_tagger","vader_lexicon","punkt"])

# PARAMETER 1
sia=SentimentIntensityAnalyzer()
nonroot_file1=pd.read_csv(r'C:/Users/aniur/Desktop/NLP REVIEW 3/Review 3/scraped_data.csv')
p=[]
n=[]
score=0
nonroot_file = nonroot_file1['Text']

for sen in nonroot_file:
  score=sia.polarity_scores(sen)
  print(score , "\n")
  # print("Pos = " , score['pos'])
  # print("Neg = " , score['neg'])
  if (score['pos']<score['neg']):
    n.append(sen)
  else:
    p.append(sen)
p_file=pd.DataFrame(p)
p_file.to_csv(r'C:/Users/aniur/Desktop/NLP REVIEW 3/Review 3/positive_data.csv')
n_file=pd.DataFrame(n)
n_file.to_csv(r'C:/Users/aniur/Desktop/NLP REVIEW 3/Review 3/negative_data.csv')

# PARAMETER 2
file=pd.read_csv(r'C:/Users/aniur/Desktop/NLP REVIEW 3/Review 3/cleaned_data.csv') # the file with root words
p_temp=pd.read_csv(r'C:/Users/aniur/Desktop/NLP REVIEW 3/Review 3/positive_data.csv')
n_temp=pd.read_csv(r'C:/Users/aniur/Desktop/NLP REVIEW 3/Review 3/negative_data.csv')
p_file = p_temp['0']
n_file = n_temp['0']
# def text_to_tokens(text_seqs):
#     token_seqs = [[word.lower_ for word in encoder(text_seq)] for text_seq in text_seqs]
#     return token_seqs
def text_to_tokens(text_seqs):
    # token_seqs = [[word.lower_ for word in encoder(text_seq)] for text_seq in text_seqs]
    for i in text_seqs:
      token_seqs = []
      x = i.lower().split()
      token_seqs.append(x)
    return token_seqs
p_token = text_to_tokens(p_file)
n_token = text_to_tokens(n_file)
#print(p_token)