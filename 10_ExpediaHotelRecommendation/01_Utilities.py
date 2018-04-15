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

def append_files(files,outfile,delimiters=[',']):
    '''
    Append together multiple files with same or different schemas.
    Usage: append_files([FileA,FileB,FileC],[DelimiterA,DelimiterB,DelimiterC],outfile)
    If all the delimtiers are same, pass only one delimiter: append_files([FileA,FileB,FileC],[Delimiter],outfile)
    Order out output columns: Sequential order of header in the input file list
    '''
    # Define Output Writer
    outfile1 = open(outfile,'wb')
    writer = csv.writer(outfile1)
    
    # Read all the headers and make a master header
    master_header = []
    if len(delimiters) <= len(files):
        delimiters = [delimiters[0] for x in files]
    for file,delimiter in zip(files,delimiters):
        with open(file,'rb') as fp:
            header = csv.reader(fp,delimiter=delimiter).next()
            master_header += filter(lambda x: x not in master_header,header)
            
    # Write out the master header
    writer.writerow(master_header)
    
    # Iteratively read the files and append them to the output file after re-ordering the columns
    for file,delimiter in zip(files,delimiters):
        with open(file,'rb') as fp:
            # Read header of the input file
            reader = csv.reader(fp,delimiter=delimiter)
            header = reader.next()
            len_header = len(header)
            header = dict(zip(header,range(len(header))))
            for row in reader:
                # Add null string for columns not present in this file
                row.append('')
                # Re-order the columns
                output_row = [row[header.get(column,len_header)] for column in master_header]
                writer.writerow(output_row)
    
# parser = argparse.ArgumentParser()
# parser.add_argument("--ifile", help="Input files split by |")
# parser.add_argument("--ofile", help="Output file")
# parser.add_argument("--d", help="Delimiters of input file split by ~. Default: Comma")
# args = parser.parse_args()

# if args.d:
    # append_files(args.ifile.split('|'),args.ofile,args.d.split('~'))
# else:
    # append_files(args.ifile.split('|'),args.ofile)
    
# files = ['/data/arpit.goel/11_WTS_Stackers/input/%d.csv'%i for i in [2016,2014,2015,2016]]
# append_files(files,[','],'/data/arpit.goel/11_WTS_Stackers/input/combined_Test.csv')

parser = argparse.ArgumentParser()
parser.add_argument("--ifile", help="Input file")
parser.add_argument("--ofile", help="Output file. Default: Input name with suffix.")
parser.add_argument("--d", help="Delimiter. Default: Comma")
parser.add_argument("--chunks", help="Number of Output Files. Default: 10")
parser.add_argument("--samplingCols", help="Sampling Columns separated by |")

args = parser.parse_args()
infile = args.ifile
delimiter = args.d if args.d else ','
outfile = args.ofile if args.ofile else None
chunks = args.chunks if args.chunks else 10
sampling_columns = args.samplingCols if args.samplingCols else []
split_file(infile,delimiter=delimiter,outfile=outfile,chunks=chunks,sampling_columns=sampling_columns)
