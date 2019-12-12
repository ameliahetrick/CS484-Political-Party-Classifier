# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:10:07 2019

@author: jedwa
Script to quickly clean primary file, get rid of moderators
"""
import csv

names = set(['Cordes', 'Dinan', 'Mitchell', 'Bash', 'Tapper', 'Regan', 
             'Lopez', 'Cramer', 'Quintanilla', 'Louis', 'UNKNOWN', 'QUESTION',
             'Ham', 'Hewitt', 'Cuomo (VIDEO)', 'Quick', 'Woodruff', 'Ifill', 
             'Cavuto', 'Muir', 'Garrett', 'Epperson', 'Harwood', 'Salinas',
             'Santelli', 'Kelly', 'Blitzer', "O'Reilly (VIDEO)", 'Tumulty', 
             'Dickerson','OTHER', 'UNKNOWN (TRANSLATED)', 'Salinas (TRANSLATED)',
             'CANDIDATES', 'MacCallum', 'Raddatz', 'Cooney', 'Hannity', 
             'Ramos (VIDEO)', 'Ramos (TRANSLATED)', 'Seib', 'QUESTION (TRANSLATED)',
             'Wallace', 'ArrarÃ¡s', 'Lemon', 'Todd', 'Baker', 'AUDIENCE', 
             'Obradovich', 'Hemmer', 'Mcelveen', 'Baier', 'Holt', 'Speaker',
             'Bartiromo', 'Levesque', 'Cooper', 'Maddow', 'Ramos', 'Strassel'])

lineNums = []
newFile = open('primary_wo_noise.csv', 'w', newline='')
with open('primary_debates_cleaned.csv', 'r') as csvfile:
    csv1 = csv.reader(csvfile, delimiter=',')
    csv2 = csv.writer(newFile, delimiter=',', quoting=csv.QUOTE_ALL)
    for row in csv1:
        if row[1] in names:
            continue
        else:
            csv2.writerow(row)
            
            

            
    




