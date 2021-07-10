from sklearn.feature_extraction.text import CountVectorizer

sentence = "It was the best of times, it was the worst of times."
print(sentence)

# preprocessing
sentence_lst = [word.lower().strip(".") for word in sentence.split()]

# look at 2 words to the left and 2 to the right of each word in sentence
context_length = 3

# iterate through preprocessed sentence, word by word, and return list of different BOW representations for the context words that surround each word
def get_cbows(sentence_lst, context_length):
  cbows = list()
  for i, val in enumerate(sentence_lst):
    if i < context_length:
      pass
    elif i < len(sentence_lst) - context_length:
      context = sentence_lst[i-context_length:i] + sentence_lst[i+1:i+context_length+1]
      vectorizer = CountVectorizer()
      vectorizer.fit_transform(context)
      context_no_order = vectorizer.get_feature_names()
      cbows.append((val,context_no_order))
  return cbows

# define cbows here:
cbows = get_cbows(sentence_lst, context_length)

# iterates through a sentence word by word and returns an ordered list of the different context words that surround each word in the sentence
def get_skip_grams(sentence_lst, context_length):
  skip_grams = list()
  for i, val in enumerate(sentence_lst):
    if i < context_length:
      pass
    elif i < len(sentence_lst) - context_length:
      context = sentence_lst[i-context_length:i] + sentence_lst[i+1:i+context_length+1]
      skip_grams.append((val, context))
  return skip_grams

# define skip_grams here:
skip_grams = get_skip_grams(sentence_lst, context_length)

try:
  # order of context words doesnt matter
  print('\nContinuous Bag of Words')
  for cbow in cbows:
    print(cbow)
except:
  pass
try:
  # order of context words is preserved
  print('\nSkip Grams')
  for skip_gram in skip_grams:
    print(skip_gram)
except:
  pass