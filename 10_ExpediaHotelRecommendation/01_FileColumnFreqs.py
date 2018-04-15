import sys,csv,os

infile=sys.argv[1]
delimiter=sys.argv[2]

def get_file_freqs(file_path):
    filename = file_path[file_path.rfind('/')+1:file_path.rfind('.')]
    if not os.path.exists(filename):
        os.mkdir(filename)
    with open(file_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        header = reader.next()
        counters = [{} for column in header]
        for row in reader:
            for i,cell in enumerate(row):
                counters[i][cell] = 1+counters[i].get(cell,0)
        for i,column in enumerate(header):
            with open('%s/%02d.%s.csv'%(filename,i,column), 'wb') as f:
                [f.write('{0},{1}\n'.format(key, value)) for key, value in counters[i].items()]
 
get_file_freqs(infile)
