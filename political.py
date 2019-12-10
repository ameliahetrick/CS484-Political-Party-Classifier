'''
put our info here

Example run:
python3 [fill this in]

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

train = ''
test = ''
output = ''

# all cli components should be present and in order, otherwise give mssg and exit
if (len(sys.argv) == 4):
    train = sys.argv[1]
    test = sys.argv[2]
    output = sys.argv[3]
else:
    print('Please enter acceptable command line arguments:')
    print('\t[program] [train in] [test in] [labels out]')
    sys.exit(0)

# print confirmations of command line inputs
print('')
print('Train:', train)
print('Test:', test)
print('Output:', output)
print('')
print('Working...')
print()

#open training file
trainfile = open(train, 'r')

#Code to read training file,




testfile = open(test, 'r')

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


