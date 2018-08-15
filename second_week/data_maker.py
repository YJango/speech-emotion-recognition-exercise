import os
import numpy as np
import pandas as pd

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
##htk file reader/writer
#htk writer
def write_htk(features,outputFileName,fs=100,dt=9):
    """
    in: file name
    out: out[0] is the data
    """
    sampPeriod = 1./fs
    pk =dt & 0x3f
    features=np.atleast_2d(features)
    if pk==0:
        features =features.reshape(-1,1)
    with open(outputFileName,'wb') as fh:
        fh.write(struct.pack(">IIHH",len(features),int(sampPeriod*1e7),features.shape[1]*4,dt))
        features=features.astype(">f")
        features.tofile(fh)
#htk reader
def read_htk(inputFileNmae, framePerSecond=100):
    """
    in: file name
    """
    kinds=['WAVEFORM','LPC','LPREFC','LPCEPSTRA','LPDELCEP','IREFC','MFCC','FBANK','MELSPEC','USER','DISCRETE','PLP','ANON','???']
    with open(inputFileNmae, 'rb') as fid:
        nf=struct.unpack(">l",fid.read(4))[0]
        fp=struct.unpack(">l",fid.read(4))[0]*-1e-7
        by=struct.unpack(">h",fid.read(2))[0]
        tc=struct.unpack(">h",fid.read(2))[0]
        tc=tc+65536*(tc<0)
        cc='ENDACZK0VT'
        nhb=len(cc)
        ndt=6
        hb=list(int(math.floor(tc * 2 ** x)) for x in range(- (ndt+nhb),-ndt+1))
        dt=tc-hb[-1] *2 **ndt
        if any([dt == x for x in [0,5,10]]):
            aise('NOt!')
        data=np.asarray(struct.unpack(">"+"f"*int(by/4)*nf,fid.read(by*nf)))
        d=data.reshape(nf,int(by/4))
    t=kinds[min(dt,len(kinds)-1)]
    return (d,fp,dt,tc,t)
def df_features(filelist):
    '''
    filelist: a list of ".txt" files that contain extracted features
    orderfile: whether return order
    return: 
            2D-array with shape: [n_example,dim_feature]
            if orderfile == True: also return a list of order of examples
    '''
    id_features_list = []
    get_columns = True
    for file in filelist:
        # example id
        fileid = file.split('/')[-1].split('.')[0]
        with open(file,'r') as f:
            lines = f.readlines()
            if get_columns:
                columnnames = [l.split('@attribute ')[-1].split(' numeric')[0] for l in lines[3:-5] if 'class numeric' not in l]
                get_columns = False
                print('dimension of features = %s' %len(columnnames))
                #print(columnnames)
            last_line = lines[-1]
            # features
            features = last_line.split(',')[1:-1]
            id_features_list.append([fileid] + features)
    df = pd.DataFrame(id_features_list,columns=['id']+columnnames)
    return df

def df_label(file):
    '''
    file: a file that id and label of examples
    return: dataframe of id and label
    
    '''
    with open(file,'r') as f:
        lines = f.readlines()
    data = [[l.split('\t')[0],l.split('\t')[1][:-1]] for l in lines]
    df = pd.DataFrame(data,columns=['id','label'])
    return df

def get_train_test_df(feature_set_name = 'IS13'):
    
    PATH = '/home/jyu/haoweilai/'
    labelfile = '/home/jyu/haoweilai/data/Train/P_Train_27KB.txt'
    # get label dataframe
    df_train_labels = df_label(labelfile)
    print('make data for %s' %feature_set_name)
    
    # get train and test file lists
    trainfiles = get_filenames(path=PATH+'extraction/train/%s/' %feature_set_name,extension='.txt')
    testfiles = get_filenames(path=PATH+'extraction/test/%s/' %feature_set_name,extension='.txt')
    
    # get dataframes for train and test
    df_train_features= df_features(trainfiles)
    df_test_features = df_features(testfiles)
    
    # convert to numeric type and merge features and label for train set
    train_data = pd.merge(df_train_features,df_train_labels).set_index('id').apply(pd.to_numeric)
    test_data = df_test_features.set_index('id').apply(pd.to_numeric)
    
    # save csv
    save_path = PATH+'dataframe/train_%s.csv' %feature_set_name
    train_data.to_csv(save_path)
    print('saved',save_path)
    save_path = PATH+'dataframe/test_%s.csv' %feature_set_name
    test_data.to_csv(save_path)
    print('saved',save_path)
    
    # compute mean and std from train set
    mean = train_data.mean()
    std = train_data.std()
    np.save(PATH+'dataframe/%s_mean_std.npy' %feature_set_name,[mean,std])
    print('saved',PATH+'dataframe/%s_mean_std.npy' %feature_set_name)
    
#get_train_test_df(feature_set_name = 'IS09')
#get_train_test_df(feature_set_name = 'IS13')
get_train_test_df(feature_set_name = 'IS10')
get_train_test_df(feature_set_name = 'IS16')