import os,sys
import subprocess
def get_filenames(path, shuffle=False, extension='.wav'):
    # get all file names 
    files= os.listdir(path) 
    filepaths = [path+file for file in files if not os.path.isdir(file) and extension in file]
    # shuffle
    if shuffle:
        ri = np.random.permutation(len(filepaths))
        filepaths = np.array(filepaths)[ri]
    #print(filepaths)
    return filepaths
    
def opensmiler(infile, outfold='IS13',config='IS13_ComParE',toolfold='/DB/Tools/opensmile-2.3.0/',extension='.txt'):
    '''
    infile: single input file to be extracted
    outfold: where to save the extracted file with the same name
    extension: ".txt" or ".csv" 
    
    '''
    # tool and config
    tool = '%sbin/linux_x64_standalone_libstdc6/SMILExtract' %toolfold
    config = '%sconfig/%s.conf' %(toolfold,config)
    
    # get infile and outfile names
    infilename = infile
    outfilename = '%s/%s%s' %(outfold, infile.split('/')[-1].split('.wav')[0], extension)
    cmd = '%s -C %s -I %s -O %s' %(tool,config,infilename,outfilename)
    
    #execute
    if subprocess.call(cmd, shell=True) ==1:
        raise TypeError('something wrong happened')
    else:
        print('Done:',cmd)
        
PATH = '/home/jyu/haoweilai/'
# get train and test file lists
trainfiles = get_filenames(path=PATH+'data/Train/Audio/',extension='.wav')
testfiles = get_filenames(path=PATH+'data/Test/Audio/',extension='.wav')

# extract IS09 and IS13 for train and test sets
for f in trainfiles:
    opensmiler(f, outfold=PATH+'extraction/train/IS09/',config='IS09_emotion')
    opensmiler(f, outfold=PATH+'extraction/train/IS13/',config='IS13_ComParE')
    opensmiler(f, outfold=PATH+'extraction/train/IS10/',config='IS10_paraling')
    opensmiler(f, outfold=PATH+'extraction/train/IS16/',config='ComParE_2016')
for f in testfiles:
    opensmiler(f, outfold=PATH+'extraction/test/IS09/',config='IS09_emotion')
    opensmiler(f, outfold=PATH+'extraction/test/IS13/',config='IS13_ComParE')
    opensmiler(f, outfold=PATH+'extraction/test/IS10/',config='IS10_paraling')
    opensmiler(f, outfold=PATH+'extraction/test/IS16/',config='ComParE_2016')
