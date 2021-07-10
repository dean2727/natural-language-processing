from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs
# import sklearn modules here:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Setting up the combined list of friends' writing samples
friends_docs = goldman_docs + henson_docs + wu_docs
# Setting up labels for your three friends
friends_labels = [1] * 154 + [2] * 141 + [3] * 166

# Print out a document from each friend:
print("~~~~~~~~~~~~~~~~~~~~~ Friend 1: Goldman ~~~~~~~~~~~~~~~~~~~~~")
print(goldman_docs[0])
print("~~~~~~~~~~~~~~~~~~~~~ Friend 2: Henson ~~~~~~~~~~~~~~~~~~~~~")
print(henson_docs[0])
print("~~~~~~~~~~~~~~~~~~~~~ Friend 3: Wu ~~~~~~~~~~~~~~~~~~~~~")
print(wu_docs[0])

# the document we need to classify
# visit https://www.gutenberg.org/ and find lines from Emma Goldman, Matthew Henson, and TingFang Wu to see how the classifier holds up
mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""

bow_vectorizer = CountVectorizer()
# train (fit) and vectorize (transform) all the friends' writing
friends_vectors = bow_vectorizer.fit_transform(friends_docs)

mystery_vector = bow_vectorizer.transform([mystery_postcard]) # must be an array

# our Naive Bayes classifier can confirm any suspicions we have about which friend wrote the mystery postcard
friends_classifier = MultinomialNB()

# Train the classifie
friends_classifier.fit(friends_vectors, friends_labels)

# make the prediction
predictions = friends_classifier.predict(mystery_vector)
# can also call predict_proba(mystery_vector) to get the probabilities for each person

# predictions just has 1 element in the array. if its empty, its none of the 3 friends
mystery_friend = predictions[0] if predictions[0] else "someone else"

print("\nThe postcard was from {}!".format(mystery_friend))