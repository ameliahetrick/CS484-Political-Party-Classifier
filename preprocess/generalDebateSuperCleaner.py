
"""
Created on Tue Dec 10 11:32 2019

@author: jedwa, modified by amelia
Script to quickly clean primary file, get rid of moderators
"""
import csv

names = set(['Clinton', 'Trump'])

lineNums = []
newFile = open('general_wo_noise.csv', 'w', newline='')
with open('general_debates.csv', 'r', encoding="latin-1") as csvfile:
    csv1 = csv.reader(csvfile, delimiter=',')
    csv2 = csv.writer(newFile, delimiter=',', quoting=csv.QUOTE_ALL)
    for row in csv1:
        if row[1] not in names:
            continue
        else:
            csv2.writerow(row)
            
            

            
    




