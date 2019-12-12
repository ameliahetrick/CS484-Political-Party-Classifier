'''
James Edwards  | jedwar30@gmu.edu
Amelia Hetrick | ahetric@gmu.edu

Created: November 2019
Updated: December 11 2019

Example run:
[program] [train in] [train labels in] [test in] [test labels in] [predicted labels out]
python3 political.py primary_stemmed.csv primaryClassified.txt general_stemmed.csv generalClassified.txt out.txt

Majority of code outline taken from:
https://www.kaggle.com/kredy10/simple-lstm-for-text-classification

Sequential model adapted from:
https://www.datatechnotes.com/2019/06/text-classification-example-with-keras.html

Keras documentation:
https://keras.io/
'''

import sys
import csv
from collections import defaultdict
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers



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



''''''


# create lists to hold data
train= []
trainLabels = []
test = []
testLables = []
output = []


# open train file
with open(trainFile, 'r') as file:
    csv1 = csv.reader(file, delimiter=',')
    columns = defaultdict(list)
    next(file)
    for row in csv1:
        for (i,v) in enumerate(row):
            columns[i].append(v)
    train = columns[2]

# open train labels file
with open(trainLabelsFile, 'r') as file:
    trainLabels = file.readlines()

# open test file
with open(testFile, 'r') as file:
    csv2 = csv.reader(file, delimiter=',')
    columns = defaultdict(list)
    next(file)
    for row in csv2:
        for (i,v) in enumerate(row):
            columns[i].append(v)
    test = columns[2]

# open test labels file
with open(testLabelsFile, 'r') as file:
    testLabels = file.readlines()


# convert data and labels to np arrays
train = np.array(train)
trainLabels = np.array(trainLabels)
test = np.array(test)

# convert test labels to integers
for el in range(len(testLabels)):
    testLabels[el] = int(testLabels[el])


''''''


X_train,X_test,Y_train,Y_test = train, test, trainLabels, testLabels

# set parameters
max_words = 1000
max_len = 300

# create tokenizer
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)

# update format of input train data to matrix
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
sequences_matrix = np.array(sequences_matrix)

# update format of input test data to matrix
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
test_sequences_matrix = np.array(test_sequences_matrix)


''''''
# baseline neural network: MLP
"""
from sklearn.neural_network import MLPClassifier 
X = sequences_matrix
y = Y_train

model = MLPClassifier()
model = model.fit(X, y)

output = model.predict(test_sequences_matrix)

for el in range(len(output)):
    if output[el] == '1\n':
        output[el] = 1
    output[el] = int(output[el])

output = [int(i) for i in output]

output = np.array(output)
testLabels = np.array(testLabels)

from sklearn.metrics import accuracy_score
print(accuracy_score(testLabels, output))

sys.exit(0)
"""
''''''


# create sequential model and add layers
vocab_size=len(tok.word_index)+1
embedding_dim=50
model=Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))

model.add(layers.LSTM(units=50,return_sequences=True))
model.add(layers.LSTM(units=10))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8))
model.add(layers.Dense(1, activation="sigmoid"))
    
model.compile(optimizer=RMSprop(), loss="binary_crossentropy", metrics=['accuracy'])

model.summary()

# fit model to train labels
model.fit(sequences_matrix,Y_train,batch_size=64,epochs=5)





''''''




# print out accuracy/loss results
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# binarize results to fall into categories of either 0 or 1
output = model.predict(test_sequences_matrix)
output[output>0.5]=1
output[output<=0.5]=0
output = output.tolist()

# write results to output file
results = open(outputFile, 'w')
i=0
for val in output:
    results.write(str(int(output[i][0]))+'\n')
    i+=1








