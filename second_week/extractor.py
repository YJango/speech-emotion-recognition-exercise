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
def opensmiler(infile, outfold='IS13',config='IS13_ComParE',toolfold='/DB/Tools/opensmile-2.3.0/',extension='.txt',lld='O'):
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
    cmd = '%s -C %s -I %s -%s %s' %(tool,config,infilename,lld,outfilename)
    
    #execute
    if subprocess.call(cmd, shell=True) ==1:
        raise TypeError('something wrong happened')
    else:
        print('Done:',cmd)
        
def mfccer(wavs, mfcc_fold):
    #make mfcc list
    mfcc_list =mfcc_fold+'mfcc.scp'
    with open(mfcc_list,'w') as f:
        for wav in wavs:
            fname = wav.split('/')[-1].split('.')[0]
            f.write('%s %s\n' %(wav,mfcc_fold+fname+'.mfc'))
    
    # tool and config
    tool = 'HCopy'
    config = mfcc_fold+'config'
    with open(config,'w') as f:
        f.write("SOURCEFORMAT = WAV\n")
        f.write("TARGETKIND = MFCC_0_D_A_Z\n")
        f.write("TARGETRATE = 100000.0\n")
        f.write("SAVEWITHCRC = T\n")
        f.write("WINDOWSIZE = 250000.0\n")
        f.write("USEHAMMING = T\n")
        f.write("PREEMCOEF = 0.97\n")
        f.write("NUMCHANS = 24\n")
        f.write("CEPLIFTER = 22\n")
        f.write("NUMCEPS = 12\n")
    # get infile and outfile names
    cmd = '%s -T 1 -C %s -S %s' %(tool,config,mfcc_list)
    #execute
    if subprocess.call(cmd, shell=True) ==1:
        raise TypeError('something wrong happened')
    else:
        print('Done:',cmd)
        

PATH = '/home/jyu/haoweilai/'
# get train and test file lists
trainfiles = get_filenames(path=PATH+'data/train/Audio/',extension='.wav')
testfiles = get_filenames(path=PATH+'data/test/Audio/',extension='.wav')
  
# extract IS09 and IS13 for train and test sets
#mfccer(trainfiles, mfcc_fold=PATH+'extraction/train/mfcc/')
#mfccer(testfiles, mfcc_fold=PATH+'extraction/test/mfcc/')

'''
for f in trainfiles:
    opensmiler(f, outfold=PATH+'extraction/train/IS09/',config='IS09_emotion')
    opensmiler(f, outfold=PATH+'extraction/train/IS13/',config='IS13_ComParE')
    opensmiler(f, outfold=PATH+'extraction/train/IS13lld/',config='IS13_ComParE',lld='D')
    opensmiler(f, outfold=PATH+'extraction/train/IS10/',config='IS10_paraling')
    opensmiler(f, outfold=PATH+'extraction/train/IS10lld/',config='IS10_paraling',lld='D')
    opensmiler(f, outfold=PATH+'extraction/train/IS16/',config='ComParE_2016')
    opensmiler(f, outfold=PATH+'extraction/train/IS16lld/',config='ComParE_2016',lld='D')
for f in testfiles:
    opensmiler(f, outfold=PATH+'extraction/test/IS09/',config='IS09_emotion')
    opensmiler(f, outfold=PATH+'extraction/test/IS13/',config='IS13_ComParE')
    opensmiler(f, outfold=PATH+'extraction/test/IS13lld/',config='IS13_ComParE',lld='D')
    opensmiler(f, outfold=PATH+'extraction/test/IS10/',config='IS10_paraling')
    opensmiler(f, outfold=PATH+'extraction/test/IS10lld/',config='IS10_paraling',lld='D')
    opensmiler(f, outfold=PATH+'extraction/test/IS16/',config='ComParE_2016')'''
    opensmiler(f, outfold=PATH+'extraction/test/IS16lld/',config='ComParE_2016',lld='D')

