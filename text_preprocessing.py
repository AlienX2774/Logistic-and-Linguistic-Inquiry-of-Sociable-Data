import re
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords
import pandas as pd
import nltk
tweets = pd.read_csv(r'C:\Users\saura\Downloads\Logistic-and-Linguistic-Inquiry-of-Socialable-Data-main\file_name.csv')
X=tweets['Text']
nltk.download('stopwords')
stop_words=stopwords.words('english')
punct=string.punctuation
stemmer=PorterStemmer()
cleaned_data=[]
for i in range(len(X)):
    tweet=re.sub('[^a-zA-Z]', ' ', X.iloc[i])
    tweet=tweet.lower().split()
    tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words) and (word not in punct)]
    tweet=' '.join(tweet)
    cleaned_data.append(tweet)
for i in range(len(cleaned_data)):
    print(cleaned_data[i])
