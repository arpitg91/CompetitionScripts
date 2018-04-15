import csv
import argparse
import random

def sample_file(infile,delimiter=',',outfile=None,sampling_rate=0.1,sampling_columns=[]):
    
    # If no outfile specified, use input file name as outfile name
    if outfile == None:
        outfile = infile[infile.rfind('/')+1:infile.rfind('.')]
        
    # Read the input file header and initialise outfiles
    infile1 = open(infile,'rb')
    outfile = open('%s_SAMPLE.TXT'%(outfile),'wb')
    reader = csv.reader(infile1,delimiter=delimiter)
    writer = csv.writer(outfile,delimiter=delimiter)
    header = reader.next()
    writer.writerow(header)
    header = dict(zip(header,range(len(header))))
        
    # If outputkeys are specified use linenum as key else make the key using specified columns
    if len(sampling_columns) == 0:
        for row in reader:
            if random.random() < sampling_rate:
                writers[reader.line_num%chunks].writerow(row)
    else:
        for row in reader:
            key = hash('|'.join([row[header[column]] for column in sampling_columns]))
            if key%100 < sampling_rate:
                writer.writerow(row)

    # Close the input and output files
    outfile.close()
    infile1.close()

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description =
'''
Sample a huge file into small file. 
Sampling Columns: Key columns on which file has to be sampled. E.g. Get all transactions of 10% of customers.
Default Parameters:
    outfile: Input file name with suffix
    sampling_rate: 0.1
    sampling_columns: None

    Example: 
        python sample_file.py --ifile file1 --ofile ofile --d '|' --sampling_rate 0.05 --samplingCols 'COL1|COL2'
'''
)

parser.add_argument("--ifile", help="Input file")
parser.add_argument("--ofile", help="Output file. Default: Input name with suffix.")
parser.add_argument("--d", help="Delimiter. Default: Comma")
parser.add_argument("--sampling_rate", help="Sampling Rate. Default: 0.1")
parser.add_argument("--samplingCols", help="Sampling Columns separated by |. Default: None")

args = parser.parse_args()
infile = args.ifile
delimiter = args.d if args.d else ','
outfile = args.ofile if args.ofile else None
sampling_rate = int(args.sampling_rate) if args.sampling_rate else 0.1
sampling_columns = args.samplingCols.split('|') if args.samplingCols else []
sample_file(infile,delimiter=delimiter,outfile=outfile,sampling_rate=sampling_rate,sampling_columns=sampling_columns)
