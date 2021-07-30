import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nltk.download('punkt')
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')

df = pd.read_csv('labeled.csv')
df['toxic'] = df['toxic'].astype('int')

snowball = SnowballStemmer(language="russian")
russian_stop_words = stopwords.words("russian")

def tokenize_(sentence: str, remove_stop_words: bool = True):
    tokens = word_tokenize(sentence, language="russian")
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens

model_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=lambda x: tokenize_(x))),
    ("model", LogisticRegression(random_state=42))
]
)
model_pipeline.fit(df["comment"], df["toxic"])


if model_pipeline.predict(['Здравтсвуй']) == 0:
    print('comment not toxic')
else:
    print('comment toxic')



