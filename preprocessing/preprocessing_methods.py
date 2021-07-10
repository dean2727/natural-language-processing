# regex for removing punctuation!
import re
# nltk preprocessing magic
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# grabbing a part of speech function:
from part_of_speech import get_part_of_speech

text = "So many squids are jumping out of suitcases these days that you can barely go anywhere without seeing one burst forth from a tightly packed valise. I went to the dentist the other day, and sure enough I saw an angry one jump out of my dentist's bag within minutes of arriving. She hardly even noticed."

# replace all phrases of non-alphabet characters with a space in text
cleaned = re.sub('\W+', ' ', text)
tokenized = word_tokenize(cleaned)

# remove prefixes/suffixes
stemmer = PorterStemmer()
stemmed = [stemmer.stem(token) for token in tokenized]

# bring words to root form, and fix any mispelled words
lemmatizer = WordNetLemmatizer()
# lemmatize() treats every word as a noun, so we need arg2 (get_part_of_speech(token)) 
# which tells our lemmatizer what part of speech the word is
lemmatized = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized]

print("Stemmed text:")
print(stemmed)
print("\nLemmatized text:")
print(lemmatized)


# ~~~~~~~~~~~~~~~~ MORE EXAMPLES ~~~~~~~~~~~~~~~
# removing noise from an HTML tag and tweet
headline_one = '<h1>Nation\'s Top Pseudoscientists Harness High-Energy Quartz Crystal Capable Of Reversing Effects Of Being Gemini</h1>'
tweet = '@fat_meats, veggies are better than you think.'
headline_no_tag = re.sub('<.?h1>', '', headline_one)
tweet_no_at = re.sub('@', '', tweet)

# tokenizing by word or sentence
ecg_text = 'An electrocardiogram is used to record the electrical conduction through a person\'s heart. The readings can be used to diagnose cardiac arrhythmias.'
tokenized_by_word = word_tokenize(ecg_text)
tokenized_by_sentence = sent_tokenize(ecg_text)
try:
  print('Word Tokenization:')
  print(tokenized_by_word)
except:
  print('Expected a variable called `tokenized_by_word`')
try:
  print('Sentence Tokenization:')
  print(tokenized_by_sentence)
except:
  print('Expected a variable called `tokenized_by_sentence`')

# upper or lower casing
my_string = 'tHiS HaS a MiX oF cAsEs' 
print(my_string.upper())
# 'THIS HAS A MIX OF CASES'
print(my_string.lower())
# 'this has a mix of cases'

# stopword removal, through word tokenization and then filtering the words if in the stopwords set
stop_words = set(stopwords.words('english')) 
nbc_statement = "NBC was founded in 1926 making it the oldest major broadcast network in the USA"
word_tokens = word_tokenize(nbc_statement) 
statement_no_stop = [word for word in word_tokens if word not in stop_words]
print(statement_no_stop)
# ['NBC', 'founded', '1926', 'making', 'oldest', 'major', 'broadcast', 'network', 'USA']

# stemming
stemmer = PorterStemmer()
populated_island = 'Java is an Indonesian island in the Pacific Ocean. It is the most populated island in the world, with over 140 million people.'
island_tokenized = word_tokenize(populated_island)
stemmed = [stemmer.stem(token) for token in island_tokenized]

# lemmatizing 
lemmatized_pos = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in island_tokenized]


