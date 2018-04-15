import numpy as np
import csv
import os
import argparse
import collections
import itertools
import random
import string

def checkNumeric(s):
    if s != '':
        try:
            float(s)
        except ValueError:
            return False
    return True

def makeTempFilename():
    chars = list(string.ascii_letters + string.digits)
    while True:
        random.shuffle(chars)
        fname = 'EDD_%s'%(''.join(chars[:8]))
        if not os.path.isfile(fname):
            return fname

def determineFieldTypes(reader, hasHeader, numLines, tmpWriter):
    print  'Determining field types...'
    
    for (ct, row) in enumerate(reader):
        if ct % 10000 == 0:
            print  '\r%i'%(ct),

        if ct == 0:
            nFields = len(row)
            if hasHeader:
                header = row
                header = ['%s_%s'%(ct+1, i) for (ct, i) in enumerate(header)]
            else:
                header = [str(i) for i in xrange(nFields)]
                tmpWriter.writerow(row)

            numericSet = set(header)
            
        else:
            tmpWriter.writerow(row)
            numericMask = (checkNumeric(s) for s in row)
            categoric = set(h for (h, val) in itertools.izip(header, numericMask) if val == False)
            numericSet -= categoric
            if ct >= numLines - 1:
                break
    print  '\r%i'%(ct)

    numericMask = [True if h in numericSet else False for h in header]
    return numericMask, header, nFields        

def updateNumStats(num, numStats):
    if num != '':
        num = float(num)
        numStats['max'] = max(num, numStats['max'])
        numStats['min'] = min(num, numStats['min'])

        numStats['count'] += 1
        delta = num - numStats['mean']
        numStats['mean'] += delta / numStats['count']
        numStats['M2'] += delta * (num - numStats['mean'])
    else:
        numStats['blank'] += 1
    return numStats

def initStats(numericMask):
    stats = [{'max' : -np.inf, 'min' : np.inf, 'count' : 0, 'mean' : 0.0, 'M2' : 0.0, 'blank' : 0} if isNum
        else collections.Counter() for isNum in numericMask]
    return stats

def processLines(lines, numericMask, header, nFields):
    print 'Analyzing lines...'
    stats = initStats(numericMask)
    for (ct, row) in enumerate(lines):
        if ct % 10000 == 0:
            print  '\r%i'%(ct),

        if len(row) != nFields:
            print  '\nWARNING: Number of headers and number of columns in row %i do not match!'%(ct)

        for (stat, isNumeric, val, field) in itertools.izip(stats, numericMask, row, header):
            if isNumeric:
                try:
                    updateNumStats(val, stat)
                except ValueError:
                    print  '\nERROR: unexpected non-numeric value in row %i, column %s. Ignoring value'%(ct, field)
            else:
                stat.update([val])
            
    return stats

def printStats(stats, numericMask, header,outfile):
    outHeaders = ['Field Number and Name', 'Type', 'Num Blanks', 'Num Entries',
        'Num Unique', 'Min', 'Max', 'Mean', 'Stddev', 'Top 10 Cat Values']

    writer = csv.DictWriter(outfile, outHeaders,lineterminator='\n')
    writer.writeheader()

    for (stat, isNumeric, field) in itertools.izip(stats, numericMask, header):
        row = {}
        row['Field Number and Name'] = field

        if isNumeric:
            row['Type'] = 'Num'
            row['Num Blanks'] = stat['blank']
            row['Num Entries'] = stat['count']
            row['Min'] = stat['min']
            row['Max'] = stat['max']
            row['Mean'] = stat['mean']
            try:
                row['Stddev'] = np.sqrt(stat['M2'] / (stat['count'] - 1))
            except ZeroDivisionError:
                row['Stddev'] = ''
        else:
            row['Type'] = 'Cat'
            row['Num Blanks'] = stat.get('', 0)
            row['Num Entries'] = sum(stat.values()) - stat.get('', 0)
            row['Num Unique'] = len(stat)
            row['Top 10 Cat Values'] = ' | '.join('%s: %i'%(cat, ct) for (cat, ct) in stat.most_common(10))
            
        writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description =
        """
            If you get and exception at "numeric = set(reader.fieldnames)" it's because
            you are not using python 2.7.  Please ask IT to update your python version.
        
            Usage: pipe a csv file into EDD.py
            Output: EDD of the data to stdout
            Example:
                python ./EDD.py -d "," -e -if infile -of outfile
        """)
    parser.add_argument('-d', '--delimiter', default = ',', dest = 'delimiter',help = 'The field delimiter')
    parser.add_argument('-e', '--has-header', default = False, action = 'store_true', dest = 'hasHeader',help = 'Flag to indicate that the first line is the header')
    parser.add_argument('-l', '--num-lines-check', default = 10000, type = int, dest = 'numLines',help = 'Number of lines to read to determine field types')
    parser.add_argument('-if', dest = 'infile',help = 'Input File Name')
    parser.add_argument('-of', default = "edd.csv", dest = 'outfile',help = 'Output File Name')
    args = parser.parse_args()
    
    reader = csv.reader(open(args.infile,"r"), delimiter = args.delimiter)

    tmpName = makeTempFilename()
    with open(tmpName, 'w') as tmpPipe:
        tmpWriter = csv.writer(tmpPipe, delimiter = args.delimiter, lineterminator = '\n')
        numericMask, header, nFields = determineFieldTypes(reader, args.hasHeader, args.numLines, tmpWriter)

    with open(tmpName, 'r') as tmpPipe:
        tmpReader = csv.reader(tmpPipe, delimiter = args.delimiter)
        stats = processLines(itertools.chain(tmpReader, reader), numericMask, header, nFields)

    os.remove(tmpName)
    if args.outfile == "edd.csv":
        outfile = open(args.infile[args.infile.replace("\\","/").rfind("/")+1:args.infile.rfind(".")]+"_edd.csv","w")
    else:
        outfile = open("edd.csv","w")
    
    printStats(stats, numericMask, header, outfile)
    outfile.close()
    
if __name__ == '__main__':
    main()