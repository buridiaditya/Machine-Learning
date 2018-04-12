import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from neuralnetwork import NeuralNetwork

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

y_target = np.zeros((N,1))
for i in range(len(y_target_names)):
	if(y_target_names[i] == "ham"):
		y_target[i] = 1

# Split training and test error
trainN = int(0.8*N)

X_train = x_encoded[0:trainN]
y_train = y_target[0:trainN]

X_test = x_encoded[trainN:]
y_test = y_target[trainN:]

# Training the model
N,M = X_train.shape
nn = NeuralNetwork(M,np.array([100,50]),1)
nn.train(X_train,y_train,X_test,y_test,epochs=100,learning_rate=1e-1,learning_rate_decay=0.98)
nn.plot()
