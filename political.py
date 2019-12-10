'''
put our info here

Example run:
python3 political.py primary_wo_noise.csv general_wo_noise.csv out.dat

'''

import sys                                                     # system args (cli)
from nltk.stem import *                                        # snowball stemmer
from sklearn.feature_extraction.text import TfidfVectorizer    # tf x idf vectorizer
from nltk.corpus import stopwords                              # stop words
from sklearn.metrics.pairwise import cosine_similarity         # cosine similarity
from scipy.sparse import csr_matrix                            # sparse matrix
from collections import Counter                                # list counter (most common element)
import csv

#imports to tokenize/stem
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re

'''
Step 0. Read command line arguments.
Looking for:
  0) file name
  1) train file
  2) test file
  3) output file (test file's classification labels)
'''

trainFile = ''
trainLabelsFile = ''
testFile = ''
testLablesFile = ''
outputFile = ''

# all cli components should be present and in order, otherwise give mssg and exit
if (len(sys.argv) == 6):
    trainFile = sys.argv[1]
    trainLabelsFile = sys.argv[2]
    testFile = sys.argv[3]
    testLabelsFile = sys.argv[4]
    outputFile = sys.argv[5]
else:
    print('Please enter acceptable command line arguments:')
    print('\t[program] [train in] [train labels] [test in] [test labels] [predicted labels out]')
    sys.exit(0)

# print confirmations of command line inputs
print('')
print('Train:', trainFile)
print('Train Labels:', trainLabelsFile)
print('Test:', testFile)
print('Test Labels:', testLabelsFile)
print('Output:', outputFile)
print('')
print('Working...')
print()

#open training file
with open(trainFile, 'r'):
    print("yay")

#Code to read training file,




#testfile = open(test, 'r')

'''
Step 1. 
'''



print('Step 1 of 3. Still working...')
print()



'''
Step 2. 
'''



print('Step 2 of 3. Still working...')
print()



'''
Step 3. 
'''



print('Step 3 of 3. Program run done! See output file.')




'''
keras
'''


'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))

model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(features_set, labels, epochs = 100, batch_size = 32)
'''







