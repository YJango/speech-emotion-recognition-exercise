import pandas as pd
import numpy as np
import data
import layers
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def model_fn(features, mode, params):
    layer =layers.basic_layer(mode)
    
    ######################### forward ###########################
    a = features['DNN']
    a0 = tf.zeros_like(a)
    a_dim = a.get_shape().as_list()[-1]
    b = features['RNN']
    b0 = tf.zeros_like(b)
    b_dim = b.get_shape().as_list()[-1]
    
    t = features['label']
    
    denses = params.denses
    # correlation loss
    def cor(a, b):
        a_mean = tf.reduce_mean(a, axis=0)
        a_centered = a - a_mean
        b_mean = tf.reduce_mean(b, axis=0)
        b_centered = b - b_mean
        corr_nr = tf.reduce_sum(a_centered * b_centered, axis=0)
        corr_dr1 = tf.reduce_sum(a_centered * a_centered, axis=0)
        corr_dr2 = tf.reduce_sum(b_centered * b_centered, axis=0)
        corr_dr = tf.sqrt(corr_dr1 * corr_dr2 + 1e-11)
        corr = corr_nr / corr_dr
        return tf.reduce_mean(corr)
  
    def cornet(a, b, reuse = tf.AUTO_REUSE):
        # a --> ha
        a2h = layer.dense_layers(a, [denses[0]], name='auto_a2h', reuse=reuse, trainable=params.h_trainable)
        # b --> hb
        b2h = layer.dense_layers(b, [denses[0]], name='auto_b2h', reuse=reuse, trainable=params.h_trainable)

        
        h = a2h + b2h
        if params.dropout!=0:
            h = tf.layers.dropout(inputs=h, rate=params.dropout , training= mode == tf.estimator.ModeKeys.TRAIN)
            print('add dropout %s' %params.dropout)
                
        h2 = layer.dense_layers(h, denses[1:], name='h2',dp=params.dropout, L2 = params.L2,reuse=reuse,trainable=~params.h_trainable)
        
        # h -->a
        h2a = layer.dense_layers(h, [a_dim], name='auto_h2a', act=tf.identity, reuse=reuse, trainable=params.h_trainable)
        # h -->b
        h2b = layer.dense_layers(h, [b_dim], name='auto_h2b', act=tf.identity, reuse=reuse, trainable=params.h_trainable)

        return {'h2a':h2a, 'h2b':h2b, 'h':h, 'h2':h2,'cor_loss':cor(a2h, b2h)}
    
    ######################### outputs ###########################
        
    out_a   = cornet(a , b0) # only input a
    out_b   = cornet(a0, b ) # only input b
    out_ab   = cornet(a, b) # only input ab

    z = tf.concat(values=[a,b],axis = -1)

    # auto
    predictions_from_a = tf.concat(values=[out_a['h2a'],out_a['h2b']],axis = -1)
    predictions_from_b = tf.concat(values=[out_b['h2a'],out_b['h2b']],axis = -1)
    predictions_from_ab = tf.concat(values=[out_ab['h2a'],out_ab['h2b']],axis = -1)

    loss_a = tf.losses.mean_squared_error(labels = z, predictions = predictions_from_a)
    loss_b = tf.losses.mean_squared_error(labels = z, predictions = predictions_from_b)
    loss_ab = tf.losses.mean_squared_error(labels = z, predictions = predictions_from_ab)

    loss_p = loss_a + loss_b + loss_ab

        
    outputs = layer.dense_layers(out_ab['h2'], [1], name='output1', act=tf.identity, reuse = tf.AUTO_REUSE)
    outputs_a = layer.dense_layers(out_a['h2'], [1], name='output1', act=tf.identity, reuse = tf.AUTO_REUSE)
    outputs_b = layer.dense_layers(out_b['h2'], [1], name='output1', act=tf.identity, reuse = tf.AUTO_REUSE)
    
    
    predictions = {"labels": t,
              "outputs": outputs,
              "outputs_a": outputs_a,
              "outputs_b": outputs_b,
              "h":out_ab['h'],
               "h2":out_ab['h2']
               }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # target losses
    mse_loss_a = tf.losses.mean_squared_error(labels = t, predictions = outputs_a, scope='mse_loss_a')
    mse_loss_b = tf.losses.mean_squared_error(labels = t, predictions = outputs_b, scope='mse_loss_b')
    mse_loss_ab = tf.losses.mean_squared_error(labels = t, predictions = outputs, scope='mse_loss_ab')
    mse_loss = mse_loss_a + mse_loss_b + mse_loss_ab
    
    # exclude variables
    trainable_variables = tf.trainable_variables()
    print([v.name for v in trainable_variables])
    
    cor_loss= out_ab['cor_loss']
    
    L2_loss = 0
    if params.L2>0:
        L2_loss = tf.losses.get_regularization_loss(scope='L2')
        
    autoencoder_loss = loss_p*(1-params.lamda) - cor_loss*params.lamda
    
    task_loss = (mse_loss_ab)*(1-params.lamda)+(mse_loss_a + mse_loss_b)*params.lamda

    if params.h_trainable ==0:
        print('task loss')
        total_loss = task_loss+L2_loss+autoencoder_loss*0
    else:
        print('cor loss')
        total_loss = autoencoder_loss+task_loss*0
    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
              loss=total_loss,
              global_step=tf.train.get_global_step(),
              learning_rate=params.learning_rate,
              optimizer="Adam",
              summaries=["learning_rate"])
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    eval_metric_ops = {
                 "rmse": tf.metrics.root_mean_squared_error(labels = t, predictions = outputs),
                 "rmse_a": tf.metrics.root_mean_squared_error(labels = t, predictions = outputs_a),
                 "rmse_b": tf.metrics.root_mean_squared_error(labels = t, predictions = outputs_b),
                 # autoencoder losses
                 "a2z_rmse": tf.metrics.root_mean_squared_error(labels = z, predictions = predictions_from_a),
                 "b2z_rmse": tf.metrics.root_mean_squared_error(labels = z, predictions = predictions_from_b),
                 "ab2z_rmse": tf.metrics.root_mean_squared_error(labels = z, predictions = predictions_from_ab),
                 }
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops)
    
    
    
def input_fn(foldID, setname, feature_name, shuffle_buffer,batchsize=2,epoch=1,path='/home/jyu/haoweilai/tfrecords/',label_in_feature=True):
    # how many shards to use
    if setname =='test':
        shards=[foldID]
    elif setname =='train':
        shards=[i for i in range(7) if i!=foldID]
    elif setname =='predict':
        shards=list(range(7))
    # label
    dataset_paths=[]
    for s in shards:
        dataset_path = '%s/%s/%s.tfrecord' %(path, 'label', s)
        dataset_paths.append(dataset_path)
    print('label',shards)
    info_path = '%s/%s_info.csv' %(path, 'label')
    dataset = data.get_dataset(paths=dataset_paths, data_info=info_path, num_parallel_calls=4, prefetch_buffer=batchsize)
    
    datasets = [dataset]
    
    # feature sets
    
    dataset_path = '/home/jyu/haoweilai/tfrecords2/DNN/%s_%s.tfrecord' %(setname,foldID)
    #dataset_paths.append(dataset_path)
    info_path = '/home/jyu/haoweilai/tfrecords/DNN_info.csv'
    dataset = data.get_dataset(paths=dataset_path, data_info=info_path, num_parallel_calls=4, prefetch_buffer=batchsize)
    datasets.append(dataset)
    
    
    dataset_path = '/home/jyu/haoweilai/tfrecords2/RNN/%s_%s.tfrecord' %(setname,foldID)
    #dataset_paths.append(dataset_path)
    info_path = '/home/jyu/haoweilai/tfrecords/RNN_info.csv'
    dataset = data.get_dataset(paths=dataset_path, data_info=info_path, num_parallel_calls=4, prefetch_buffer=batchsize)
    datasets.append(dataset)
    

    dataset = tf.data.Dataset.zip(tuple(datasets))
    if shuffle_buffer>1:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.batch(batchsize)
    dataset = dataset.repeat(epoch)
    iterator = dataset.make_one_shot_iterator()
    example = iterator.get_next()
    
    features={}
    
    features['DNN']=example[1]['DNN']
    features['RNN']=example[2]['RNN']
        
    if label_in_feature:
        print('label in features')
        features['label'] =example[0]['label']
        return features
    else:
        return (features,example[0]['label'])
    
def setparam():
        model_params = tf.contrib.training.HParams(
                                foldID=foldID,
                                feature_name = feature_name,
                                denses = denses,
                                L2 = L2,
                                batchsize =batchsize,
                                dropout = dropout,
                                lamda = lamda,
                                h_trainable = h_trainable,
                                learning_rate = learning_rate)
        return model_params
logpath = '/home/jyu/haoweilai/cor2'
foldID=0
feature_name = ['']
denses = [512,512,512,512]
L2 = 0
batchsize = 2
dropout = 0.2
learning_rate = 1e-4
Epoch=10
n_example=1200
lamda = 0.2
h_trainable = 1
for denses in [[1024,128]]:
    for batchsize in [16]:
        for learning_rate in [5e-4]:
            rmses=[]
            pccs=[]
            for foldID in range(7):
                # model path
                L2 = 0
                dropout = 0
                learning_rate = 1e-4
                Epoch=5
                n_example=1200
                lamda = 0.2
                h_trainable = 1
                model_params = setparam()
                
                path_name_list = (model_params.foldID,''.join([str(r) for r in feature_name]),''.join([str(r) for r in model_params.denses]),model_params.L2,model_params.dropout,model_params.batchsize,model_params.learning_rate)
                path_name = logpath+"/id%s_f%s_d%s_L2%s_dp%s_b%s_l%s" %path_name_list
                # train
                model_params = setparam()
                myconfig = tf.estimator.RunConfig(
                                        model_dir = path_name,
                                        save_summary_steps=int(n_example/model_params.batchsize),
                                        save_checkpoints_steps=int(n_example/model_params.batchsize),
                                        save_checkpoints_secs=None,
                                        session_config=None,
                                        keep_checkpoint_max=1,
                                        keep_checkpoint_every_n_hours=int(n_example/model_params.batchsize),
                                        log_step_count_steps=int(n_example/model_params.batchsize))
                regressor = tf.estimator.Estimator(
                            model_fn = model_fn,
                            config = myconfig,
                            params=model_params)
                train_spec = tf.estimator.TrainSpec(lambda: input_fn(foldID, 'train', feature_name, n_example,batchsize,1), max_steps=int(n_example/batchsize)*Epoch)
                eval_spec = tf.estimator.EvalSpec(lambda: input_fn(foldID, 'test', feature_name, 1,batchsize=batchsize))
                tf.estimator.train_and_evaluate(regressor, train_spec, eval_spec)
                
                
                ###############################################
                # model path
                path_name_list = (model_params.foldID,''.join([str(r) for r in feature_name]),''.join([str(r) for r in model_params.denses]),model_params.L2,model_params.dropout,model_params.batchsize,model_params.learning_rate)
                path_name = logpath+"/id%s_f%s_d%s_L2%s_dp%s_b%s_l%s" %path_name_list
                
                # params
                Epoch=10
                h_trainable = 0
                learning_rate = 1e-4
                L2 = 0
                dropout = 0
                lamda = 0.1
                # train
                model_params = setparam()
                myconfig = tf.estimator.RunConfig(
                                        model_dir = path_name,
                                        save_summary_steps=int(n_example/model_params.batchsize),
                                        save_checkpoints_steps=int(n_example/model_params.batchsize),
                                        save_checkpoints_secs=None,
                                        session_config=None,
                                        keep_checkpoint_max=1,
                                        keep_checkpoint_every_n_hours=int(n_example/model_params.batchsize),
                                        log_step_count_steps=int(n_example/model_params.batchsize))
                regressor = tf.estimator.Estimator(
                            model_fn = model_fn,
                            config = myconfig,
                            params=model_params)
                train_spec = tf.estimator.TrainSpec(lambda: input_fn(foldID, 'train', feature_name, n_example,batchsize,1), max_steps=int(n_example/batchsize)*Epoch)
                eval_spec = tf.estimator.EvalSpec(lambda: input_fn(foldID, 'test', feature_name, 1,batchsize=batchsize))
                tf.estimator.train_and_evaluate(regressor, train_spec, eval_spec)
                
                
                ###############################################
                predictions = list(regressor.predict(input_fn=lambda:input_fn(foldID, 'test', feature_name, 1)))
                P = np.array([p['outputs'] for p in predictions])
                L = np.array([p['labels'] for p in predictions])
                rmse=np.sqrt(mean_squared_error(L,P))
                pcc=np.sqrt(r2_score(L,P))
                rmses.append(rmse)
                pccs.append(pcc)
                print('rmse: %s | pcc: %s' %(rmse, pcc))
            with open(logpath+'/eval.txt','a') as f:
                f.write('%s %s %s\n' %(path_name,np.array(rmses).mean(),np.array(pccs).mean()))