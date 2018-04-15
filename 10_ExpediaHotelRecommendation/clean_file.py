import csv, sys

infile = sys.argv[1]
outfile = sys.argv[2]
delimiter=sys.argv[3]

infile = open(infile,'rb')
outfile = open(outfile,'wb')
reader = csv.reader(infile,delimiter=delimiter)
writer = csv.writer(outfile,delimiter=delimiter)

header = reader.next()
writer.write(map(lambda x: x.upper().replace('  ','').replacce(' ','_'),header))

for row in reader:
    row = map(lambda x: x.replace(delimiter,'').strip().replace('\r',''),row)
    if len(row) == len(header):
        writer.write(row)
        
infile.close()
outfile.close()