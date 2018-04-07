import numpy as np
import re
import nltk
nltk.download()
from nltk.corpus import stopwords

# Read file in lines
fd = open("Assignment_2_data.txt","r")
file_data = fd.read()
lines = (file_data).splitlines()

y_target_names = []
x = []

stop_words = set(stopwords.words('english'))
# Pasing input file into labels and features
for i in range(len(lines)):
	words = re.split('\W+',lines[i])
	words = list(filter(lambda temp: (temp != '' ),words))
	words = [w for w in words if not w in stop_words]
	#words = lines[i].split()
	y_target_names.append(words[0])
	x.append(words[1:])
	#print (x[i])
	
# Creating One hot encoding of the data
tokens = set([val for sublist in x for val in sublist])
print (tokens)


# assuming one hot encoding available
N, M = X.shape




