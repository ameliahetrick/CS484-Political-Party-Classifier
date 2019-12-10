# -*- coding: utf-8 -*-
"""
this is for making labels from general data (+1, -1)
"""

# dont forget to manually delete first row (column labels)

'''
# PRIMARY
import csv

names = set(['Democratic'])

newFile = open('primaryClassified.txt', 'w', newline='')
with open('primary_stemmed.csv', 'r') as csvfile:
    csv1 = csv.reader(csvfile, delimiter=',')
    for row in csv1:
        if row[4] in names:
            newFile.write('1\n')
        else:
            newFile.write('0\n')
'''

# GENERAL
import csv

names = set(['Clinton'])

newFile = open('generalClassified.txt', 'w', newline='')
with open('general_stemmed.csv', 'r') as csvfile:
    csv1 = csv.reader(csvfile, delimiter=',')
    for row in csv1:
        if row[1] in names:
            newFile.write('1\n')
        else:
            newFile.write('0\n')
