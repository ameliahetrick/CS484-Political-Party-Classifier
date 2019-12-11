'''
James Edwards
Amelia Hetrick

Created: November 2019
Updated: December 10 2019

Example run:
python3 political.py primary_stemmed.csv primaryClassified.txt general_stemmed.csv generalClassified.txt out.txt

Majority of code outline taken from:
https://www.kaggle.com/kredy10/simple-lstm-for-text-classification
https://www.datatechnotes.com/2019/06/text-classification-example-with-keras.html

https://keras.io/ for extras
'''

import sys                                                     # system args (cli)
from nltk.stem import *                                        # snowball stemmer
from sklearn.feature_extraction.text import TfidfVectorizer    # tf x idf vectorizer
from nltk.corpus import stopwords                              # stop words
from sklearn.metrics.pairwise import cosine_similarity         # cosine similarity
from scipy.sparse import csr_matrix                            # sparse matrix
from collections import Counter                                # list counter (most common element)
import csv
from collections import defaultdict
import numpy as np

#imports to tokenize/stem
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re

'''
Read command line arguments.
Looking for:
  0) input  file  name
  1) input  train data   file
  2) input  train labels file
  3) input  test  data   file
  4) input  test  labels file (ground truth)
  5) output test  labels file (predicted)
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
    print('\t[program] [train in] [train labels in] [test in] [test labels in] [predicted labels out]')
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


train= []
trainLabels = []
test = []
testLables = []
output = []


#open train file
with open(trainFile, 'r') as file:
    csv1 = csv.reader(file, delimiter=',')
    columns = defaultdict(list)
    next(file)
    for row in csv1:
        for (i,v) in enumerate(row):
            columns[i].append(v)
    train = columns[2]
train = np.array(train)
#print(train[0], type(train[0]))

#open train labels file
with open(trainLabelsFile, 'r') as file:
    trainLabels = file.readlines()

#open test file
with open(testFile, 'r') as file:
    csv2 = csv.reader(file, delimiter=',')
    columns = defaultdict(list)
    next(file)
    for row in csv2:
        for (i,v) in enumerate(row):
            columns[i].append(v)
    test = columns[2]
test = np.array(test)


#open test labels file
with open(testLabelsFile, 'r') as file:
    testLabels = file.readlines()

for el in range(len(testLabels)):
    testLabels[el] = int(testLabels[el])



#sys.exit(0)




'''
keras
'''


'''
#https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()


#model.add(LSTM(units=50, return_sequences=True, input_shape=(train.shape, 1)))
model.add(LSTM(units=50, return_sequences=True, input_shape=(train.shape)))
#model.add(LSTM(units=50, return_sequences=True, input_dim=(3)))
#max_talk_len = 500
#X_train = sequence.pad_sequence(train, maxlen=max_talk_len)

#model.add(Embedding(500, embedding_vecor_length, input_length=max_talk_len))


model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#model.compile(loss='binary_crossentropy', optimizer = 'adam')

model.fit(train, trainLabels, epochs = 100, batch_size = 32)
#model.fit(X_train, trainLabels, epochs = 100, batch_size = 32)
'''



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

#matplotlib inline

df = pd.read_csv(trainFile,delimiter=',',encoding='latin-1')
df.head()

print('\n')

cols = [0,1,3,4,5,6]
df.drop(df.columns[cols],axis=1,inplace=True)
df.head()
df.info()

#Labels = pd.DataFrame({'trainLabels': trainLabels})

'''
X = df.Text
Y = trainLabels
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
'''

X_train,X_test,Y_train,Y_test = train, test, trainLabels, testLabels
X_train,X_test,Y_train,Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

#max_words = 1000
max_words = 500
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
sequences_matrix = np.array(sequences_matrix)
print(sequences_matrix)

def RNN():
    '''inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)'''

    vocab_size=len(tok.word_index)+1
    embedding_dim=50
    model=Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    model.add(layers.LSTM(units=50,return_sequences=True))
    model.add(layers.LSTM(units=10))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", 
         metrics=['accuracy'])
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])


model.fit(sequences_matrix,Y_train,batch_size=32,epochs=1)#,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])


test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
test_sequences_matrix = np.array(test_sequences_matrix)

accr = model.evaluate(test_sequences_matrix,Y_test)


print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

output = model.predict(test_sequences_matrix)
output[output>0.5]=1
output[output<=0.5]=0
output = output.tolist()
#list(output)
print(output[0:10])


output2 = []
for el in range(len(output)):
    output2.append(int(output[el][0]))
print(output2[0:10])

print(type(testLabels[0]), type(output2[0]))
from sklearn.metrics import accuracy_score
print(accuracy_score(testLabels, output2))



# note to selves: get rid of output and change to putput2 V

with open(outputFile, 'w') as file:
    file.write('\n'.join(map(str, output2)))
'''results = open(outputFile, 'w')
i=0
for val in output:
    results.write(str(int(output[i][0]))+'\n')
    i+=1
'''







