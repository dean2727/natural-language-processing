'''
In this project, I use term frequency-inverse document frequency (tf-idf) to analyze each
article’s content and uncover the terms that best describe each article, providing quick insight
into each article’s topic.
'''

import language_models.preprocessing
import pandas as pd
import numpy as np
from articles import articles
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# view article
print(articles[0])

# preprocess articles
processed_articles = [preprocess_text(s) for s in articles]
print(processed_articles)

# initialize and fit CountVectorizer
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(processed_articles)

transformer = TfidfTransformer(norm=None)

# convert counts to tf-idf
tfidf_scores_transformed = transformer.fit_transform(counts)

# initialize and fit TfidfVectorizer
vectorizer = TfidfVectorizer(norm=None)
tfidf_scores = vectorizer.fit_transform(processed_articles)

# check if tf-idf scores are equal
# CountVectorizer + TfidfTransformer == TfidfVectorizer?
if np.allclose(tfidf_scores_transformed.todense(), tfidf_scores.todense()):
  print(pd.DataFrame({'Are the tf-idf scores the same?':['YES']}))
else:
  print(pd.DataFrame({'Are the tf-idf scores the same?':['No, something is wrong :(']}))

# get vocabulary of terms
try:
  feature_names = vectorizer.get_feature_names()
except:
  pass

# get article index
try:
  article_index = [f"Article {i+1}" for i in range(len(articles))]
except:
  pass

# create pandas DataFrame with word counts
try:
  df_word_counts = pd.DataFrame(counts.T.todense(), index=feature_names, columns=article_index)
  print(df_word_counts)
except:
  pass

# create pandas DataFrame(s) with tf-idf scores
try:
  df_tf_idf = pd.DataFrame(tfidf_scores_transformed.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

try:
  df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

# get highest scoring tf-idf term for each article (naive way to get the topic of the article)
for i in range(1, 11):
  # print the index of the term with the highest tf-idf score for Articles 1 to 10
# .idxmax() is a helpful tool for returning the index of the highest value in a DataFrame column
  print(df_tf_idf[[f'Article {i}']].idxmax())


