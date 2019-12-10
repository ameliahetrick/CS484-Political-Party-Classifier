# -*- coding: utf-8 -*-
"""
this is for making labels from general data (+1, -1)
"""

import csv

names = set(['Democratic'])

newFile = open('primaryClassified.dat', 'w', newline='')
with open('primary_stemmed.csv', 'r') as csvfile:
    csv1 = csv.reader(csvfile, delimiter=',')
    for row in csv1:
        if row[4] in names:
            newFile.write('+1\n')
        else:
            newFile.write('-1\n')
