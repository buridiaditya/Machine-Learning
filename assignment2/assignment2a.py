import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Read file in lines
fd = open("Assignment_2_data.txt","r")
file_data = fd.read()
lines = (file_data).splitlines()

ps = PorterStemmer()

y_target_names = []
x = []

stop_words = set(stopwords.words('english'))
unique_tokens = set()
# Pasing input file into labels and features 
for i in range(len(lines)): 
	# Splitting based on white space and non alpha numberic characters
	words_tokens = re.split('\W+',lines[i])
	words_tokens = list(filter(lambda temp: (temp != '' ),words_tokens)) 
	
	# Removing stop word from the data
	filtered_sentence = []
	for w in words_tokens:
		if w not in stop_words:
		# Applying porter stemmer
			stemmed_word = ps.stem(w).lower()
			filtered_sentence.append(stemmed_word)
			

	# Store pre processed data 
	y_target_names.append(filtered_sentence[0])
	x.append(filtered_sentence[1:])
	unique_tokens.update(filtered_sentence[1:])
#print (y_target_names)
	
# One hot encoding.Note: Does not give importance to the order in which the words occur.
N = len(x)
unique_tokens = list(unique_tokens)
encoding_length = (len(unique_tokens))
x_encoded = np.zeros( ( N , encoding_length) )

# Binary encoding 
for i in range(N):
	for j in x[i]:
		ind = unique_tokens.index(j)
		x_encoded[i][ind] = 1;

y_target = np.zeros(N)
for i in range(len(y_target_names)):
	if(y_target_names[i] == "ham"):
		y_target[i] = 1

# Training the model

