import csv
import argparse

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
    
parser = argparse.ArgumentParser()
parser.add_argument("--ifile", help="Input files split by |")
parser.add_argument("--ofile", help="Output file")
parser.add_argument("--d", help="Delimiters of input file split by ~. Default: Comma")
args = parser.parse_args()

if args.d:
    append_files(args.ifile.split('|'),args.ofile,args.d.split('~'))
else:
    append_files(args.ifile.split('|'),args.ofile)
    
