from ..preprocessing import preprocess_text
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# sample documents
# change these to see what happens to the cell values
document_1 = "This is a sample sentence!"
document_2 = "I love to code."
document_3 = "Jesus loves you!"

# corpus of documents
corpus = [document_1, document_2, document_3]

# preprocess documents
processed_corpus = [preprocess_text(doc) for doc in corpus]

# initialize and fit TfidfVectorizer
vectorizer = TfidfVectorizer(norm=None)
tf_idf_scores = vectorizer.fit_transform(processed_corpus)

# get vocabulary of terms
feature_names = vectorizer.get_feature_names()
corpus_index = [n for n in processed_corpus]

# create pandas DataFrame with tf-idf scores
df_tf_idf = pd.DataFrame(tf_idf_scores.T.todense(), index=feature_names, columns=corpus_index)
print(df_tf_idf)