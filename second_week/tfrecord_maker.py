import pandas as pd
import numpy as np
import data
import tensorflow as tf
import struct
import math
import scipy.io.wavfile as wav
import speechpy
import os
from python_speech_features import fbank,mfcc
def mfcc_extractor(file_name):
    fs, signal = wav.read(file_name)
    ############# Extract logenergy features #############
    myfb=mfcc(signal,samplerate=fs,winlen=0.025,winstep=0.01,numcep=13,
                 nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
     ceplifter=22,appendEnergy=True)
    logenergy = myfb
    #logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                 #num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)
    logenergy_feature_cube_cmvn = speechpy.processing.cmvn(logenergy_feature_cube.reshape(-1,39), variance_normalization=True)
    #print('logenergy_feature_cube_cmvn shape=', logenergy_feature_cube_cmvn.shape)
    return logenergy, logenergy_feature_cube_cmvn
def fb_extractor(file_name):
    fs, signal = wav.read(file_name)
    ############# Extract logenergy features #############
    myfb,mye =fbank(signal,samplerate=fs,winlen=0.025,winstep=0.01,
      nfilt=40,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)
    logenergy = np.hstack((myfb,mye.reshape((-1,1))))
    #logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                 #num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)
    logenergy_feature_cube_cmvn = speechpy.processing.cmvn(logenergy_feature_cube.reshape(-1,123), variance_normalization=True)
    #print('logenergy_feature_cube_cmvn shape=', logenergy_feature_cube_cmvn.shape)
    return logenergy, logenergy_feature_cube_cmvn

def read_lld(fname):
    with open(fname,'r') as f:
        lines=f.readlines()
    matrix=[]
    for l in lines:
        if 'unknown' in l:
            matrix.append(l[:-1].split(';')[2:])
    return np.array(matrix,dtype='float32')

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

def get_data(setname='train',feature_set_name='IS13'):
    epsilon = 1e-15
    PATH = '/home/jyu/haoweilai/'
    df_train = pd.read_csv(PATH+'dataframe/%s_%s.csv' %(setname,feature_set_name)) 
    mean,std = np.load(PATH+'dataframe/%s_mean_std.npy' %feature_set_name)
    
    if setname =='train':
        data = (df_train.set_index('id')-mean)/(std+epsilon)
    else:
        data = (df_train.set_index('id')-mean[:-1])/(std[:-1]+epsilon)
    return data
def save_tfrecord(X, X_info, X_path):
    '''
    X: data
    X_info: how
    X_path: where
    
    '''
    writer = tf.python_io.TFRecordWriter(X_path)
    for i in np.arange(len(X)):
        features = {}
        # if the rank of X > 1, save shape as well
        if len(X[0].shape)>1:
            data.feature_writer(X_info.iloc[1], X[i].shape, features)
        data.feature_writer(X_info.iloc[0], X[i].reshape(-1).astype('float32'), features)
        tf_features = tf.train.Features(feature= features)
        tf_example = tf.train.Example(features = tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
    writer.close()
    
    
PATH = '/home/jyu/haoweilai/'
#cs_ids = setsplit(k_fold = 7)
#np.save('/home/jyu/haoweilai/dataframe/cv_ids.npy',cs_ids)
cs_ids = np.load('/home/jyu/haoweilai/dataframe/cv_ids.npy')


################################################### 'IS09','IS10','IS13','IS16'

for feature_set_name in ['IS09','IS10','IS13','IS16']:
    #####
    # predict data
    test = np.array(get_data(setname='test',feature_set_name=feature_set_name))
    print(feature_set_name, 'predict',test.shape)
    X_info = pd.DataFrame({'name':[feature_set_name],
                           'type':['float32'],
                           'shape':[(test.shape[-1],)],
                           'isbyte':[False],
                           "length_type":['fixed'],
                           "default":[np.NaN]})
    X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
    X_info.to_csv(X_info_path,index=False)
    X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,'predict')
    save_tfrecord(test, X_info, X_path)
    #####
    # train data
    for foldID in range(7):
        train = get_data(setname='train',feature_set_name=feature_set_name)
        # label
        label = np.array(train.loc[cs_ids[foldID][1],'label'])
        #print(feature_set_name, label.shape)
        X_info = pd.DataFrame({'name':['label'],
                       'type':['float32'],
                       'shape':[(1,)],
                       'isbyte':[False],
                       "length_type":['fixed'],
                       "default":[np.NaN]})
        X_info.to_csv(PATH+'/tfrecords/label_info.csv',index=False)
        X_path = PATH+'/tfrecords/label/%s.tfrecord' %foldID
        save_tfrecord(label, X_info, X_path)
        # features
        del train['label']
        shard = np.array(train.loc[cs_ids[foldID][1]])
        print(feature_set_name, 'shard %s' %foldID, shard.shape)
        X_info = pd.DataFrame({'name':[feature_set_name],
                               'type':['float32'],
                               'shape':[(shard.shape[-1],)],
                               'isbyte':[False],
                               "length_type":['fixed'],
                               "default":[np.NaN]})
        X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
        X_info.to_csv(X_info_path,index=False)
        X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,foldID)
        save_tfrecord(shard, X_info, X_path)

        
        
# MFCC39 with HTK

feature_set_name ='mfcc'
setname='train'
for foldID in range(7):
    shard = []
    for fname in cs_ids[foldID][1]:
        mfcc_file_name = PATH+'extraction/%s/%s/%s.mfc' %(setname,feature_set_name,fname)
        #print(mfcc_file_name)
        shard.append(read_htk(mfcc_file_name)[0])
    print(feature_set_name, 'shard %s' %foldID, len(shard))
    X_info = pd.DataFrame({'name':[feature_set_name,feature_set_name+'_shape'],
                           'type':['float32','int64'],
                           'shape':[(39,),(2,)],
                           'isbyte':[False,False],
                           "length_type":['var','fixed'],
                           "default":[np.NaN,np.NaN]})
    X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
    X_info.to_csv(X_info_path,index=False)
    X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,foldID)
    save_tfrecord(shard, X_info, X_path)
    
setname='test'
shard = []
ids = get_data(setname='test',feature_set_name='IS10').index
for fname in ids:
    mfcc_file_name = PATH+'extraction/%s/%s/%s.mfc' %(setname,feature_set_name,fname)
    #print(mfcc_file_name)
    mfcc = read_htk(mfcc_file_name)[0]
    mfcc = (mfcc-mfcc.mean())/mfcc.std()
    shard.append(mfcc)
print(feature_set_name, 'predict',len(shard))
X_info = pd.DataFrame({'name':[feature_set_name,feature_set_name+'_shape'],
                       'type':['float32','int64'],
                       'shape':[(39,),(2,)],
                       'isbyte':[False,False],
                       "length_type":['var','fixed'],
                       "default":[np.NaN,np.NaN]})
X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
X_info.to_csv(X_info_path,index=False)
X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,'predict')
save_tfrecord(shard, X_info, X_path)

################################################### fb+e

feature_set_name ='fb'
setname='train'
for foldID in range(7):
    shard = []
    for fname in cs_ids[foldID][1]:
        mfcc_file_name = PATH+'data/%s/Audio/%s.wav' %(setname,fname)
        #print(mfcc_file_name)
        shard.append(fb_extractor(mfcc_file_name)[1])
    print(feature_set_name, 'shard %s' %foldID, len(shard))
    X_info = pd.DataFrame({'name':[feature_set_name,feature_set_name+'_shape'],
                           'type':['float32','int64'],
                           'shape':[(120,),(2,)],
                           'isbyte':[False,False],
                           "length_type":['var','fixed'],
                           "default":[np.NaN,np.NaN]})
    X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
    X_info.to_csv(X_info_path,index=False)
    X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,foldID)
    save_tfrecord(shard, X_info, X_path)
    
setname='test'
shard = []
ids = get_data(setname='test',feature_set_name='IS10').index

for fname in ids:
    mfcc_file_name = PATH+'data/%s/Audio/%s.wav' %(setname,fname)
    #print(mfcc_file_name)
    shard.append(fb_extractor(mfcc_file_name)[1])
print(feature_set_name, 'predict',len(shard))
X_info = pd.DataFrame({'name':[feature_set_name,feature_set_name+'_shape'],
                       'type':['float32','int64'],
                       'shape':[(120,),(2,)],
                       'isbyte':[False,False],
                       "length_type":['var','fixed'],
                       "default":[np.NaN,np.NaN]})
X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
X_info.to_csv(X_info_path,index=False)
X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,'predict')
save_tfrecord(shard, X_info, X_path)



################################################### mfcc

feature_set_name ='mfcc'
setname='train'
for foldID in range(7):
    shard = []
    for fname in cs_ids[foldID][1]:
        mfcc_file_name = PATH+'data/%s/Audio/%s.wav' %(setname,fname)
        #print(mfcc_file_name)
        shard.append(mfcc_extractor(mfcc_file_name)[1])
    print(feature_set_name, 'shard %s' %foldID, len(shard))
    X_info = pd.DataFrame({'name':[feature_set_name,feature_set_name+'_shape'],
                           'type':['float32','int64'],
                           'shape':[(120,),(2,)],
                           'isbyte':[False,False],
                           "length_type":['var','fixed'],
                           "default":[np.NaN,np.NaN]})
    X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
    X_info.to_csv(X_info_path,index=False)
    X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,foldID)
    save_tfrecord(shard, X_info, X_path)
    
setname='test'
shard = []
ids = get_data(setname='test',feature_set_name='IS10').index

for fname in ids:
    mfcc_file_name = PATH+'data/%s/Audio/%s.wav' %(setname,fname)
    #print(mfcc_file_name)
    shard.append(mfcc_extractor(mfcc_file_name)[1])
print(feature_set_name, 'predict',len(shard))
X_info = pd.DataFrame({'name':[feature_set_name,feature_set_name+'_shape'],
                       'type':['float32','int64'],
                       'shape':[(120,),(2,)],
                       'isbyte':[False,False],
                       "length_type":['var','fixed'],
                       "default":[np.NaN,np.NaN]})
X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
X_info.to_csv(X_info_path,index=False)
X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,'predict')
save_tfrecord(shard, X_info, X_path)


################################################### IS10 lld

feature_set_name ='IS10lld'
setname='train'
for foldID in range(7):
    shard = []
    for fname in cs_ids[foldID][1]:
        mfcc_file_name = PATH+'extraction/%s/%s/%s.txt' %(setname,feature_set_name,fname)
        mfcc = read_lld(mfcc_file_name)
        mfcc = (mfcc-mfcc.mean(axis=0))/(mfcc.std(axis=0)+1e-15)
        shard.append(mfcc)
    print(feature_set_name, 'shard %s' %foldID, len(shard))
    X_info = pd.DataFrame({'name':[feature_set_name,feature_set_name+'_shape'],
                           'type':['float32','int64'],
                           'shape':[(77,),(2,)],
                           'isbyte':[False,False],
                           "length_type":['var','fixed'],
                           "default":[np.NaN,np.NaN]})
    X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
    X_info.to_csv(X_info_path,index=False)
    X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,foldID)
    save_tfrecord(shard, X_info, X_path)
    
setname='test'
shard = []
ids = get_data(setname='test',feature_set_name='IS10').index

for fname in ids:
    mfcc_file_name = PATH+'extraction/%s/%s/%s.txt' %(setname,feature_set_name,fname)
    mfcc = read_lld(mfcc_file_name)
    mfcc = (mfcc-mfcc.mean(axis=0))/(mfcc.std(axis=0)+1e-15)
    shard.append(mfcc)
print(feature_set_name, 'predict',len(shard))
X_info = pd.DataFrame({'name':[feature_set_name,feature_set_name+'_shape'],
                       'type':['float32','int64'],
                       'shape':[(77,),(2,)],
                       'isbyte':[False,False],
                       "length_type":['var','fixed'],
                       "default":[np.NaN,np.NaN]})
X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
X_info.to_csv(X_info_path,index=False)
X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,'predict')
save_tfrecord(shard, X_info, X_path)

################################################### IS13 lld

feature_set_name ='IS13lld'
setname='train'
for foldID in range(7):
    shard = []
    for fname in cs_ids[foldID][1]:
        mfcc_file_name = PATH+'extraction/%s/%s/%s.txt' %(setname,feature_set_name,fname)
        mfcc = read_lld(mfcc_file_name)
        mfcc = (mfcc-mfcc.mean(axis=0))/(mfcc.std(axis=0)+1e-15)
        shard.append(mfcc)
    print(feature_set_name, 'shard %s' %foldID, len(shard))
    X_info = pd.DataFrame({'name':[feature_set_name,feature_set_name+'_shape'],
                           'type':['float32','int64'],
                           'shape':[(77,),(2,)],
                           'isbyte':[False,False],
                           "length_type":['var','fixed'],
                           "default":[np.NaN,np.NaN]})
    X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
    X_info.to_csv(X_info_path,index=False)
    X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,foldID)
    save_tfrecord(shard, X_info, X_path)
    
setname='test'
shard = []
ids = get_data(setname='test',feature_set_name='IS10').index

for fname in ids:
    mfcc_file_name = PATH+'extraction/%s/%s/%s.txt' %(setname,feature_set_name,fname)
    mfcc = read_lld(mfcc_file_name)
    mfcc = (mfcc-mfcc.mean(axis=0))/(mfcc.std(axis=0)+1e-15)
    shard.append(mfcc)
print(feature_set_name, 'predict',len(shard))
X_info = pd.DataFrame({'name':[feature_set_name,feature_set_name+'_shape'],
                       'type':['float32','int64'],
                       'shape':[(77,),(2,)],
                       'isbyte':[False,False],
                       "length_type":['var','fixed'],
                       "default":[np.NaN,np.NaN]})
X_info_path = PATH+'/tfrecords/%s_info.csv' %(feature_set_name)
X_info.to_csv(X_info_path,index=False)
X_path = PATH+'/tfrecords/%s/%s.tfrecord' %(feature_set_name,'predict')
save_tfrecord(shard, X_info, X_path)