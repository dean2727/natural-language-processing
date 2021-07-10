import os
import gensim
import spacy
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words

# get list of all speech files
files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])
#print(files)

# read each speech file
speeches = [read_file(file) for file in files]

# preprocess each speech (list of strings -> list of lists, where each list is the words in a sentence)
processed_speeches = process_speeches(speeches)

# merge speeches (remove outer brackets)
all_sentences = merge_speeches(processed_speeches)

# view most frequently used words
most_freq_words = most_frequent_words(all_sentences)
#print(most_freq_words)

# create gensim model of all speeches
all_prez_embeddings = gensim.models.Word2Vec(all_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom
similar_to_freedom = all_prez_embeddings.most_similar("freedom", topn=20)
#print(similar_to_freedom)

# Let’s train a word embedding model on a single president and see how their word embeddings differ from the collection of all presidents.
# given a president’s name, return a list of processed sentences from every inaugural address given by the president.
roosevelt_sentences = get_president_sentences("franklin-d-roosevelt")

# view most frequently used words of Roosevelt
roosevelt_most_freq_words = most_frequent_words(roosevelt_sentences)
#print(roosevelt_most_freq_words)

# create gensim model for Roosevelt
roosevelt_embeddings = gensim.models.Word2Vec(roosevelt_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for Roosevelt
roosevelt_similar_to_freedom = roosevelt_embeddings.most_similar("freedom", topn=20)
# results are different here, since its just 1 document, so the tf-idf score is very low, so the common words will take over. in other words, our dataset is limited, so we get generalizable word embeddings
#print(roosevelt_similar_to_freedom)

# get processed sentences of multiple presidents
rushmore_prez_sentences = get_presidents_sentences(["washington","jefferson","lincoln","theodore-roosevelt"])

# view most frequently used words of presidents
rushmore_most_freq_words = most_frequent_words(rushmore_prez_sentences)
#print(rushmore_most_freq_words)

# create gensim model for the presidents
rushmore_embeddings = gensim.models.Word2Vec(rushmore_prez_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for presidents
rushmore_similar_to_freedom = rushmore_embeddings.most_similar("freedom", topn=20)
# a little less generalizable word embeddings, but still generalizable
#print(rushmore_similar_to_freedom)


'''
Who are your favorite presidents? Do you want to see how their choice of words compares to presidents you view less favorably? Or controversial presidents?

Perhaps you want to analyze presidents by political party affiliation?

Create a new word embedding model trained on a corpus of sentences from the speeches of a selection of presidents that you decide. Find the words used similarly to “freedom”, and explore the embeddings of other words in the corpus.

How do the embeddings from your model compare to the other models you have built? What surprises do you find?
'''