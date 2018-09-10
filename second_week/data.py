import pandas as pd
import numpy as np
import tensorflow as tf

import os

def get_filenames(path,shuffle=False, extension='.tfrecord'):
    # get all file names with the extension
    files= os.listdir(path) 
    filepaths = [path+file for file in files if not os.path.isdir(file) and extension in file]
    # shuffle
    if shuffle:
        ri = np.random.permutation(len(filepaths))
        filepaths = np.array(filepaths)[ri]
    return filepaths

def feature_writer(df, value, features):
    '''
    Writes a single feature in features
    Args:
        value: an array : the value of the feature to be written

    Note:
        the tfrecord type will be as same as the numpy dtype
        if the feature's rank >= 2, the shape (type: int64) will also be added in features
        the name of shape info is: name+'_shape'
    Raises:
        TypeError: Type is not one of ('int64', 'float32')
    '''
    name = df['name']
    isbyte = df['isbyte']
    length_type = df['length_type']
    default = df['default']
    dtype = df['type']
    shape = df['shape']

    # get the corresponding type function
    if isbyte:
        feature_typer = lambda x : tf.train.Feature(bytes_list = tf.train.BytesList(value=[x.tostring()]))
    else:
        if dtype == 'int64' or dtype == np.int64:
            feature_typer = lambda x : tf.train.Feature(int64_list = tf.train.Int64List(value=x))
        elif dtype == 'float32' or dtype == np.float32:
            feature_typer = lambda x : tf.train.Feature(float_list = tf.train.FloatList(value=x))
        else:
            raise TypeError("Type is not one of 'np.int64', 'np.float32'")
    # check whether the input is (1D-array)
    # if the input is a scalar, convert it to list
    if len(shape)==0 and isbyte==False:
        features[name] = feature_typer([value])
    elif len(shape)==1:
        features[name] = feature_typer(value)
    # if # if the rank of input array >=2, flatten the input and save shape info
    elif len(shape) >1:
        features[name] = feature_typer(value.reshape(-1))
        # write shape info
        features['%s_shape' %name] = tf.train.Feature(int64_list=tf.train.Int64List(value=shape))
    return features

def data_info_fn(one_example,length_type):
    # get feature information form on example
    data_info = pd.DataFrame(columns=['name','type','shape','isbyte','length_type','default'])
    i = 0
    for key in one_example:
        value = one_example[key]
        dtype = value.dtype
        shape = value.shape
        if len(shape)>1:
            data_info.loc[i] = {'name':key,
                        'type':dtype,
                        'shape':shape,
                        'isbyte':True,
                        'length_type': 'fixed',
                        'default':np.NaN}
            i+=1
            data_info.loc[i] = {'name':key+'_shape',
                        'type':'int64',
                        'shape':(len(shape),),
                        'isbyte':False,
                        'length_type':'fixed',
                        'default':np.NaN}
            i+=1
        else:
            data_info.loc[i] = {'name':key,
                        'type':dtype,
                        'shape':shape,
                        'isbyte':False,
                        'length_type': length_type,
                        'default':np.NaN}
            i+=1
    return data_info

def create_parser(data_info, retrieve_shape):

    names = data_info['name']
    types = data_info['type']
    shapes = data_info['shape']
    isbytes = data_info['isbyte']
    defaults = data_info['default']
    length_types = data_info['length_type']

    def parser(example_proto):
        def specify_features():
            specified_features = {}
            for i in np.arange(len(names)):
                # which type
                if isbytes[i]:
                    t = tf.string
                    s = ()
                else:
                    if types[i]=='uint8':
                        types[i]=tf.uint8
                    t = types[i]
                    s = shapes[i]
                # has default_value?
                if defaults[i] == np.NaN:
                    d = np.NaN
                else:
                    d = defaults[i]
                # length varies
                if length_types[i] =='fixed':
                    specified_features[names[i]] = tf.FixedLenFeature(s, t)
                elif length_types[i] =='var':
                    specified_features[names[i]] = tf.VarLenFeature(t)
                else:
                    raise TypeError("length_type is not one of 'var', 'fixed'")
            return specified_features


        # decode each parsed feature and reshape
        def decode_reshape():
            # store all decoded&shaped features
            final_features = {}
            for i in np.arange(len(names)):
                # exclude shape info
                if '_shape' not in names[i]:
                    # decode
                    if isbytes[i]:
                        # from byte format
                        decoded_value = tf.decode_raw(parsed_example[names[i]], types[i])
                        decoded_value = tf.cast(decoded_value, tf.float32)
                    else:
                        # Varlen value needs to be converted to dense format
                        if length_types[i] == 'var':
                            decoded_value = tf.sparse_tensor_to_dense(parsed_example[names[i]])
                        else:
                            decoded_value = parsed_example[names[i]]
                    # reshape
                    if '%s_shape' %names[i] in parsed_example.keys():
                        tf_shape = parsed_example['%s_shape' %names[i]]
                        decoded_value = tf.reshape(decoded_value, tf_shape)
                    final_features[names[i]] = decoded_value
                elif retrieve_shape:
                    final_features[names[i]] = parsed_example[names[i]]
            return final_features


        # create a dictionary to specify how to parse each feature 
        specified_features = specify_features()
        # parse all features of an example
        parsed_example = tf.parse_single_example(example_proto, specified_features)
        final_features = decode_reshape()

        return final_features
    return parser


def get_dataset(paths, data_info, retrieve_shape = False, num_parallel_calls=4, prefetch_buffer=2):

    filenames = paths
    
    data_info = pd.read_csv(data_info,dtype={'isbyte':bool})
    data_info['shape']=data_info['shape'].apply(lambda s: [int(i) for i in s[1:-1].split(',') if i !=''])

    #print(data_info)
    dataset = tf.data.TFRecordDataset(filenames)
    parse_function = create_parser(data_info, retrieve_shape)
    dataset = dataset.map(parse_function, num_parallel_calls=num_parallel_calls).prefetch(prefetch_buffer)
    return dataset