import pandas as pd
import numpy as np
import re
import contractions
import warnings
from collections import defaultdict
import pickle

from bs4 import BeautifulSoup

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.tokenize import word_tokenize

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

warnings.filterwarnings("ignore")

def getRidOfNonAlphabet(s):
    return re.sub(r"[^a-zA-Z]+", ' ', s)

def getRidOfHTML(s):
    return BeautifulSoup(s, "lxml").text

def getRidOfURL(s):
    return re.sub(r'http\S+', '', s)

def contractions(s):
    # Also gets rid of extra spaces
    import contractions
    ans = []
    for word in s.split():
        ans.append(contractions.fix(word))
    return ' '.join(ans)

def getRidOfStopWords(s):
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(s)
    filteredWords = []
    for word in words:
        if word not in stopWords:
            filteredWords.append(word)
    return ' '.join(filteredWords)

def lemattize(s):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(s)
        
    filteredWords = []
    for word in words:
        filteredWords.append(lemmatizer.lemmatize(word))
    return ' '.join(filteredWords)

def lemattizeWithDict(l):
    lemmatizer = WordNetLemmatizer()

    tags = defaultdict(lambda : wn.NOUN)
    tags['J'] = wn.ADJ
    tags['V'] = wn.VERB
    tags['R'] = wn.ADV

    words = []
    for word, t in pos_tag(l.split()):
        if word not in stopwords.words('english'):
            words.append(lemmatizer.lemmatize(word, tags[t[0]]))
    return str(' '.join(words))

""" Reading the raw file - Not needed once finaldf.pkl is created """
base_path = ""
df = pd.read_csv(base_path + "amazon_reviews_us_Beauty_v1_00.tsv.gz", compression='gzip', header=0,sep='\t', quotechar='"', on_bad_lines='skip')

parsed_df = df[["star_rating", "review_headline", "review_body"]]
parsed_df.dropna()

class1_df = parsed_df.loc[parsed_df['star_rating'].isin([1,2])].sample(20000)
class2_df = parsed_df.loc[parsed_df['star_rating'] == 3].sample(20000)
class3_df = parsed_df.loc[parsed_df['star_rating'].isin([4,5])].sample(20000)

class1_df["class"] = 1
class2_df["class"] = 2
class3_df["class"] = 3

final_df = pd.concat([class1_df, class2_df, class3_df])

final_df['review_headline'] = final_df['review_headline'].apply(str)
final_df['review_body'] = final_df['review_body'].apply(str)

# final_df.to_pickle("./finaldf.pkl")  

""" Data Formatting - Not needed once formatted_df.pkl is created """
# final_df = pd.read_pickle("./finaldf.pkl")

final_df['review'] = final_df[['review_headline', 'review_body']].agg(' '.join, axis=1)

review_headline_avg_before = (final_df['review_headline'].str.len()).mean()
review_body_avg_before = (final_df['review_body'].str.len()).mean()
review_avg_before = (final_df['review'].str.len()).mean()

print("Data Cleaning")
print("Before - Review Headline Avg Character Count", review_headline_avg_before)
print("Before - Review Body Avg Character Count", review_body_avg_before)
print("Before - Total Review Avg Character Count", review_avg_before)

final_df = final_df.drop('star_rating', axis=1)
final_df = final_df.drop('review_headline', axis=1)
final_df = final_df.drop('review_body', axis=1)

final_df["review"] = final_df["review"].str.lower()

final_df["review"] = final_df["review"].apply(getRidOfURL).apply(getRidOfHTML).apply(contractions).apply(getRidOfNonAlphabet)

review_avg_after = (final_df['review'].str.len()).mean()
print("After - Total Review Avg Character Count", review_avg_after)

""" Data Formatting - Not needed once formatted_df.pkl is created """
print("Data Formatting")
print("Before - Cleaned Review Avg Character Count", review_avg_after)

# final_df["review"] = final_df["review"].apply(getRidOfStopWords)
final_df["formatted_review"] = final_df["review"].apply(lemattize)

formatted_review_count = (final_df['formatted_review'].str.len()).mean()
print("After - Cleaned Review Avg Character Count", formatted_review_count)

""" Splitting """
train_x, test_x, train_y, test_y = train_test_split(final_df['formatted_review'], final_df['class'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,3))
vectorizer.fit(final_df['formatted_review'])
train_tfidf = vectorizer.transform(train_x)
test_tfidf = vectorizer.transform(test_x)

Encoder = LabelEncoder()
train_result = Encoder.fit_transform(train_y)
test_result = Encoder.fit_transform(test_y)

""" MODELS """
print("Perceptron")
perceptron = Perceptron(random_state=486)
perceptron.fit(train_tfidf, train_result)
result = perceptron.predict(test_tfidf)
report = classification_report(test_result, result, output_dict=True)

precision = (report["0"]["precision"] + report["1"]["precision"] + report["2"]["precision"])/3
recall = (report["0"]["recall"] + report["1"]["recall"] + report["2"]["recall"])/3
f1Score = (report["0"]["f1-score"] + report["1"]["f1-score"] + report["2"]["f1-score"])/3

print(report["0"]["precision"],",",report["0"]["recall"],",",report["0"]["f1-score"])
print(report["1"]["precision"],",",report["1"]["recall"],",",report["1"]["f1-score"])
print(report["2"]["precision"],",",report["2"]["recall"],",",report["2"]["f1-score"])
print(precision,",",recall,",",f1Score)

print("SVM")
model = svm.LinearSVC()
model.fit(train_tfidf, train_result)
result = model.predict(test_tfidf)
report = classification_report(test_result, result, output_dict=True)

precision = (report["0"]["precision"] + report["1"]["precision"] + report["2"]["precision"])/3
recall = (report["0"]["recall"] + report["1"]["recall"] + report["2"]["recall"])/3
f1Score = (report["0"]["f1-score"] + report["1"]["f1-score"] + report["2"]["f1-score"])/3
print(report["0"]["precision"],",",report["0"]["recall"],",",report["0"]["f1-score"])
print(report["1"]["precision"],",",report["1"]["recall"],",",report["1"]["f1-score"])
print(report["2"]["precision"],",",report["2"]["recall"],",",report["2"]["f1-score"])
print(precision,",",recall,",",f1Score)

print("Logistic Regression")
model = LogisticRegression()
model.fit(train_tfidf, train_result)
result = model.predict(test_tfidf)
report = classification_report(test_result, result, output_dict=True)

precision = (report["0"]["precision"] + report["1"]["precision"] + report["2"]["precision"])/3
recall = (report["0"]["recall"] + report["1"]["recall"] + report["2"]["recall"])/3
f1Score = (report["0"]["f1-score"] + report["1"]["f1-score"] + report["2"]["f1-score"])/3
print(report["0"]["precision"],",",report["0"]["recall"],",",report["0"]["f1-score"])
print(report["1"]["precision"],",",report["1"]["recall"],",",report["1"]["f1-score"])
print(report["2"]["precision"],",",report["2"]["recall"],",",report["2"]["f1-score"])
print(precision,",",recall,",",f1Score)

print("Multinomial Naive Bayes")
model = naive_bayes.MultinomialNB()
model.fit(train_tfidf, train_result)
result = model.predict(test_tfidf)
report = classification_report(test_result, result, output_dict=True)

precision = (report["0"]["precision"] + report["1"]["precision"] + report["2"]["precision"])/3
recall = (report["0"]["recall"] + report["1"]["recall"] + report["2"]["recall"])/3
f1Score = (report["0"]["f1-score"] + report["1"]["f1-score"] + report["2"]["f1-score"])/3
print(report["0"]["precision"],",",report["0"]["recall"],",",report["0"]["f1-score"])
print(report["1"]["precision"],",",report["1"]["recall"],",",report["1"]["f1-score"])
print(report["2"]["precision"],",",report["2"]["recall"],",",report["2"]["f1-score"])
print(precision,",",recall,",",f1Score)