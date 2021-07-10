import re
from nltk import pos_tag
from nltk import RegexpParser, Tree
from chunk_counters import np_chunk_counter

# compile(): given a regular expression pattern, compile the pattern into a regular expression object
regular_expression_object = re.compile("[A-Za-z]{4}")

# match(): given a string, looks for a single match to the regular expression that starts at the BEGINNING
# of the string. returns a match object which has the piece of text the regex matched, and at what index
# the match starts and ends. if no match, returns None
result = regular_expression_object.match("Toto")
# to access the matched text
print(result.group(0))
# if you use a regex containing capture groups, can access these groups by calling .group(<capture group #>)

# combine compile() and match() like so:
result = re.match("[A-Za-z]{4}","Toto")


# search(): looks left to right through the string to find the match and return a match object
# returns None if no match found
result = re.search("\w{8}","Are you a Munchkin?")  # match() here would return None, but not with search()


# findall(): given a regex (arg 1) and a string (arg 2), find all non-overlapping matches of the regex expression
text = "Everything is green here, while in the country of the Munchkins blue was the favorite color. But the people do not seem to be as friendly as the Munchkins, and I'm afraid we shall be unable to find a place to pass the night."
list_of_matches = re.findall("\w{8}",text)  # find all non-overlapping sequences of 8 character words
# returns ['Everythi', 'Munchkin', 'favorite', 'friendly', 'Munchkin']


# POS tagging
# pos_tag(): given a list of words that appear in the sentence, returns a list of (word, POS tag) tuples
word_sentence = ['do', 'you', 'suppose', 'oz', 'could', 'give', 'me', 'a', 'heart', '?']
part_of_speech_tagged_sentence = pos_tag(word_sentence)
# returns [('do', 'VB'), ('you', 'PRP'), ('suppose', 'VB'), ('oz', 'NNS'), ('could', 'MD'), ('give', 'VB'), ('me', 'PRP'), ('a', 'DT'), ('heart', 'NN'), ('?', '.')]
# POS tags: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html


# chunking
pos_tagged_sentence = [('where', 'WRB'), ('is', 'VBZ'), ('the', 'DT'), ('emerald', 'JJ'), ('city', 'NN'), ('?', '.')]
chunk_grammar = "AN: {<JJ><NN>}"  # adjective-noun chunk grammar
chunk_parser = RegexpParser(chunk_grammar)
# parse(): given a sentence, use the chunk grammar to chunk the sentence
scaredy_cat = chunk_parser.parse(pos_tagged_sentence)
print(scaredy_cat)
# prettier printing
Tree.fromstring(str(scaredy_cat)).pretty_print()

# NP-chunk grammar here (nouns)
chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"  # optional determiner, 0 or more adjectives, and a noun
# (or, for a VP-chunk grammar (verbs))
# chunk_grammar = "VP: {<VB.*><DT>?<JJ>*<NN><RB.?>?}"  # verb, noun phrase, optional adverb
# chunk_grammar = "VP: {<DT>?<JJ>*<NN><VB.*><RB.?>?}"  # noun phrase, verb, optional adverb
chunk_parser = RegexpParser(chunk_grammar)
# create a list to hold noun-phrase chunked sentences
np_chunked_oz = list()
# create a for loop through each pos-tagged sentence in pos_tagged_oz here (pos_tagged_oz would be a list of 
# POS tagged sentences) 
for sentence in pos_tagged_oz:
  # chunk each sentence and append to np_chunked_oz here
  np_chunked_oz.append(chunk_parser.parse(sentence))
  
# store and print the most common np-chunks here
# np_chunk_counter returns 30 most common NP-chunks from a list of chunked sentences
most_common_np_chunks = np_chunk_counter(np_chunked_oz)
print(most_common_np_chunks)

# define chunk grammar to chunk an entire sentence together
grammar = "Chunk: {<.*>+}"
parser = RegexpParser(grammar)
chunked_dancers = parser.parse(pos_tagged_sentence)
print(chunked_dancers)
# define noun phrase chunk grammar using chunk filtering here
chunk_grammar = """NP: {<.*>+}
                       }<VB.?|IN>+{"""
chunk_parser = RegexpParser(chunk_grammar)
filtered_dancers = chunk_parser.parse(pos_tagged_sentence)
print(filtered_dancers)
Tree.fromstring(str(filtered_dancers)).pretty_print()