# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:53:01 2019

@author: jedwa
PreProcessing file
"""

import csv

#imports to tokenize/stem
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re


def preProcess(file):
    
    tokenizer = RegexpTokenizer('\s+', gaps=True)
    
    stemmer = PorterStemmer()
    
    i=0
    t1 = ""
    t2 = ""
    t3 = ""
    #erase punctuation, tokenize, stem
    newFile = open("stemmed.csv", "w", newline='')
    with file as csvfile:
        csvRead = csv.reader(csvfile, delimiter=',')
        csvWrite = csv.writer(newFile, delimiter=",", quoting=csv.QUOTE_ALL)
        for row in csvRead:
            t1 = re.sub('[^A-Za-z0-9+-]+', ' ', row[2])
            t2 = tokenizer.tokenize(t1) 
            t3 = [stemmer.stem(word) for word in t2]
            row[2] = " ".join(t3)
            csvWrite.writerow(row)
            i+=1
            
    return 
            
trainfile = open("primary_wo_noise.csv", 'r')
preProcess(trainfile)
