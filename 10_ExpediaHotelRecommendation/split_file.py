import csv
import argparse

def split_file(infile,delimiter=',',outfile=None,chunks=10,sampling_columns=[]):
    '''
    Split a huge file into smaller files keeping all entries of particular columns in a single file. Uses single pass of the huge file.
    Output files will be placed in the same directory where the program is called.
    Default Parameters:
        outfile: Input file name with suffix
        chunks: 10
        sampling_columns: None
    '''
    
    # If no outfile specified, use input file name as outfile name
    if outfile == None:
        outfile = infile[infile.rfind('/')+1:infile.rfind('.')]
    else:
        outfile = outfile[:outfile.rfind('.')]
        
    # Read the input file header and initialise outfiles
    infile1 = open(infile,'rb')
    outfiles = [open('%s%02d.TXT'%(outfile,i),'wb') for i in range(chunks)]
    reader = csv.reader(infile1,delimiter=delimiter)
    writers = [csv.writer(outfile,delimiter=delimiter) for outfile in outfiles]
    header = reader.next()
    [writer.writerow(header) for writer in writers]
    header = dict(zip(header,range(len(header))))
        
    # If outputkeys are specified use linenum as key else make the key using specified columns
    if len(sampling_columns) == 0:
        for row in reader:
            writers[reader.line_num%chunks].writerow(row)
    else:
        for row in reader:
            key = hash('|'.join([row[header[column]] for column in sampling_columns]))
            writers[key%chunks].writerow(row)

    # Close the input and output files
    [outfile.close() for outfile in outfiles]
    infile1.close()

parser = argparse.ArgumentParser()
parser.add_argument("--ifile", help="Input file")
parser.add_argument("--ofile", help="Output file. Default: Input name with suffix.")
parser.add_argument("--d", help="Delimiter. Default: Comma")
parser.add_argument("--chunks", help="Number of Output Files. Default: 10")
parser.add_argument("--samplingCols", help="Sampling Columns separated by |. Default: None")

args = parser.parse_args()
infile = args.ifile
delimiter = args.d if args.d else ','
outfile = args.ofile if args.ofile else None
chunks = int(args.chunks) if args.chunks else 10
sampling_columns = args.samplingCols.split('|') if args.samplingCols else []
split_file(infile,delimiter=delimiter,outfile=outfile,chunks=chunks,sampling_columns=sampling_columns)
