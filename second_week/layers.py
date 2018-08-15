import tensorflow as tf
import numpy as np
#from ind_rnn_cell import IndRNNCell
class basic_layer(object):
    def __init__(self, mode):
        self.mode = mode
        #dense_layers
        #embedding_layer
        #rnn_layers
        #birnn_layers
        #conv1d_layer
        #downsampling1d
        #conv2d_layer
        #downsampling2d
        #cnn1d_weighted_cnn1d_layer
        #cnn2d_weighted_cnn2d_layer
        #rnn_weighted_cnn1d_layer
        #rnn_weighted_cnn2d_layer
        #rnn_indweighted_cnn1d_layer
        #rnn_indweighted_cnn2d_layer
    def activation(self, act):
        if act =='tanh':
            return tf.nn.tanh
        elif act =='sigmoid':
            return tf.nn.sigmoid
        elif act =='relu':
            return tf.nn.relu
        elif act =='linear':
            return tf.identity
        elif act =='softmax':
            return tf.nn.softmax
        
    def dense_layers(self, x, sizes, name, L2 = 0, dp = 0,
              act='relu', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0, tf.float32)):
        
        if L2 == 0:
            L2 =None
        else:
            L2 = tf.contrib.layers.l2_regularizer(L2)
            
        with tf.variable_scope(name, reuse):
            for i,size in enumerate(sizes):
                x = tf.layers.dense(inputs = x, 
                               units = size, 
                               kernel_initializer = W_init,
                               bias_initializer = b_init,
                               activation = self.activation(act),
                               name = 'dense%s_%s' %(i, size),
                               trainable = trainable,
                               kernel_regularizer = L2,
                               reuse=reuse)
                print(name+':add dense_%s' %size)
                if dp!=0:
                    x = tf.layers.dropout(inputs=x, rate=dp , training= self.mode == tf.estimator.ModeKeys.TRAIN)
                    print(name+':add dropout %s' %dp)
        return x  
    
    def embedding_layer(self, x, dim, str2id_path, embed_path, UNK_path, name = 'embedding',trainable=False, num_trainable_words=7):
        # string to index lookup table
        with tf.variable_scope(name):
            word2index_table = tf.contrib.lookup.index_table_from_file(vocabulary_file = str2id_path, num_oov_buckets=1)
            # [batch size, num_word]
            ids = word2index_table.lookup(x)
            # [batch size, num_word]
            # make embedding matrix
            load_embedding = np.load(embed_path)
            # most words
            embedding_matrix = tf.Variable(load_embedding[num_trainable_words:], trainable = trainable , name='nontrainable')
            # boundaries
            sos_soe = tf.Variable(load_embedding[:num_trainable_words],trainable = True , name='trainable')
            # zero padding
            padding_word = tf.Variable(tf.zeros([1, dim]),trainable = False , name='padding_word')
            # UNK works
            UNK_vector = tf.Variable(np.load(UNK_path),trainable = True, name='UNK_word')
            # all
            embeddings = tf.concat(values=[sos_soe, embedding_matrix, padding_word, UNK_vector], axis=0, name='embed_matrix')
            text_vectors = tf.nn.embedding_lookup(embeddings, ids, name='embedding_lookup')
            # [batch size, num_word, dim]
        return text_vectors

    def rnn_layers(self, x, sizes, name, ctype='gru', act='tanh', dp = 0):
        if ctype == "lstm":
            cells = [tf.contrib.rnn.LSTMCell(size,initializer=tf.orthogonal_initializer,activation=self.activation(act)) for size in sizes]
        elif ctype == "gru":
            cells = [tf.contrib.rnn.GRUCell(size,kernel_initializer=tf.orthogonal_initializer,activation=self.activation(act)) for size in sizes]
        elif ctype == 'ind':
            cells = [IndRNNCell(size, activation=self.activation(act)) for size in sizes]
        if dp > 0.0:
            cells = [tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=1-dp if self.mode == tf.estimator.ModeKeys.TRAIN else 1) for cell in cells]
        cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        # create RNN layer
        outputs, _ = tf.nn.dynamic_rnn(cell=cells,
                             inputs=x,
                             dtype=tf.float32,
                             scope=name+'_rnns')
        print(name+':add %s_%s' %(ctype,sizes))
        return outputs
            
    def birnn_layers(self, x, sizes, name, ctype='gru', act='tanh', dp = 0):
        if ctype == "lstm":
            cells_fw = [tf.contrib.rnn.LSTMCell(size,initializer=tf.orthogonal_initializer,activation=self.activation(act)) for size in sizes]
            cells_bw = [tf.contrib.rnn.LSTMCell(size,initializer=tf.orthogonal_initializer,activation=self.activation(act)) for size in sizes]
        elif ctype == "gru":
            cells_fw = [tf.contrib.rnn.GRUCell(size,kernel_initializer=tf.orthogonal_initializer,activation=self.activation(act)) for size in sizes]
            cells_bw = [tf.contrib.rnn.GRUCell(size,kernel_initializer=tf.orthogonal_initializer,activation=self.activation(act)) for size in sizes]
        elif ctype == 'ind':
            cells_fw = [IndRNNCell(size, activation=self.activation(act)) for size in sizes]
            cells_bw = [IndRNNCell(size, activation=self.activation(act)) for size in sizes]
        if dp > 0.0:
            cells_fw = [tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=1-dp if self.mode == tf.estimator.ModeKeys.TRAIN else 1) for cell in cells_fw]
            cells_bw = [tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=1-dp if self.mode == tf.estimator.ModeKeys.TRAIN else 1) for cell in cells_bw]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                            cells_fw=cells_fw,
                            cells_bw=cells_bw,
                            inputs=x,
                            dtype=tf.float32,
                            scope=name+'_rnns')
        print(name+':add bi%s_%s' %(ctype,sizes))
        return outputs
    
    def conv1d_layer(self, x, ksize, fsize, name, bn=False, padding='same', strides=1, L2 = 0, dp = 0,
              act='relu', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32)):
        
        if L2 == 0:
            L2 =None
        else:
            L2 = tf.contrib.layers.l2_regularizer(L2)

        x = tf.layers.conv1d(inputs = x, 
                       filters = fsize, 
                       kernel_size = ksize,
                       kernel_initializer = W_init,
                       bias_initializer = b_init,
                       activation = tf.identity,
                       padding=padding,
                       strides=strides,
                       name = name+'_conv1d_%s_%s' %(ksize,fsize),
                       kernel_regularizer = L2,
                       reuse = reuse)

        if bn==True:
            x = tf.contrib.layers.layer_norm(x)
            #x = tf.layers.batch_normalization(x, axis=[0,1],training=(self.mode == tf.estimator.ModeKeys.TRAIN))
            print(name+':add conv1d_bn_%s_%s' %(ksize,fsize))
        else:
            print(name+':add conv1d_%s_%s' %(ksize,fsize))
        x=self.activation(act)(x)
        if dp!=0:
            x = tf.layers.dropout(inputs=x, rate=dp , training= self.mode == tf.estimator.ModeKeys.TRAIN)
            print(name+':add dropout %s' %dp)
        return x  
    
    def downsampling1d(self, x, name, ptype='avg', psize=2, strides=2):
        if 'max' in ptype:
            pool = tf.layers.max_pooling1d
            print(name+':downsampling1D maxpool_size_%s_stride_%s' %(psize, strides))
        elif 'avg' in ptype:
            pool = tf.layers.average_pooling1d
            print(name+':downsampling1D avgpool_size_%s_stride_%s' %(psize, strides))
        else:
            raise TypeError("wrong downsampling1D type, should be max or avg")
        
        return pool(inputs = x, pool_size=psize, strides=strides, name = name+'downsample1D_%s_%s' %(psize, strides))
    
    def atrous_conv1d_layer(self, x, ksize, fsize, name, bn=False, padding='same', rate=1,  L2 = 0, dp = 0,
              act='relu', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32)):
        
        if L2 == 0:
            L2 =None
        else:
            L2 = tf.contrib.layers.l2_regularizer(L2)
          
        with tf.variable_scope(name, reuse = reuse):
            W =tf.get_variable(name='atrous_%s_conv1d_%s_%s_W' %(rate,ksize,fsize), shape = [ksize, x.get_shape().as_list()[-1], fsize],
                         initializer = W_init)
            b = tf.get_variable(name='atrous_%s_conv1d_%s_%s_b' %(rate,ksize,fsize), shape = [fsize],
                         initializer = b_init)
             
        #x = tf.nn.atrous_conv2d(value=x, filters=W, rate=rate, padding=padding.upper()) + b
        x = tf.nn.bias_add(tf.nn.convolution(x, W, padding=padding.upper(), dilation_rate=np.broadcast_to(rate, (1,)), name=None), b)
        if bn==True:
            x = tf.contrib.layers.layer_norm(x)
            #x = tf.layers.batch_normalization(x, training=(self.mode == tf.estimator.ModeKeys.TRAIN))
            print(name+':add atrous_%s_conv1d_bn_%s_%s' %(rate,ksize,fsize))
        else:
            print(name+':add atrous_%s_conv1d_%s_%s' %(rate,ksize,fsize))
        x=self.activation(act)(x)
        if dp!=0:
            x = tf.layers.dropout(inputs=x, rate=dp , training= self.mode == tf.estimator.ModeKeys.TRAIN)
            print(name+':add dropout %s' %dp)
        return x  
    
    
    def conv2d_layer(self, x, ksize, fsize, name, bn=False, padding='same', strides=1, L2 = 0, dp = 0,
              act='relu', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32)):
        
        if L2 == 0:
            L2 =None
        else:
            L2 = tf.contrib.layers.l2_regularizer(L2)

        x = tf.layers.conv2d(inputs = x, 
                       filters = fsize, 
                       kernel_size = ksize,
                       kernel_initializer = W_init,
                       bias_initializer = b_init,
                       activation = tf.identity,
                       padding=padding,
                       strides=strides,
                       name = name+'_conv2d_%s_%s' %(ksize,fsize),
                       kernel_regularizer = L2,
                       reuse = reuse)
        if bn==True:
            x = tf.layers.batch_normalization(x, training=(self.mode == tf.estimator.ModeKeys.TRAIN))
            print(name+':add conv2d_bn_%s_%s' %(ksize,fsize))
        else:
            print(name+':add conv2d_%s_%s' %(ksize,fsize))
        x=self.activation(act)(x)
        if dp!=0:
            x = tf.layers.dropout(inputs=x, rate=dp , training= self.mode == tf.estimator.ModeKeys.TRAIN)
            print(name+':add dropout %s' %dp)
        return x  
    
    
    def downsampling2d(self, x, name, ptype='avg', psize=2, strides=2):
        if 'max' in ptype:
            pool = tf.layers.max_pooling2d
            print(name+':downsampling2D maxpool_size_%s_stride_%s' %(psize, strides))
        elif 'avg' in ptype:
            pool = tf.layers.average_pooling2d
            print(name+':downsampling2D avgpool_size_%s_stride_%s' %(psize, strides))
        else:
            raise TypeError("wrong downsampling2D type, should be max or avg")
        
        return pool(inputs = x, pool_size=psize, strides=strides, name = name+'downsample2D')
    
    def globalpool(self, x, name, ptype='avg', axis = 1):
        if ptype =='avg':
            pool = tf.reduce_mean
        elif ptype =='sum':
            pool = tf.reduce_sum
        elif ptype =='max':
            pool = tf.reduce_max
        else:
            raise TypeError("wrong globalpool type, should be max or avg or sum")
        print(name+':%s globalpool on axis_%s' %(ptype, axis))
            
        return pool(input_tensor = x, axis=axis, name = name+'%s_globalpool_axis_%s' %(ptype, axis))
    
    def cnn1d_weighted_cnn1d_layer(self, x, cksize, wksize, fsize, name, bn=False, cstrides=1, wstrides=1, padding='same', L2 = 0, dp = 0,
              act='tanh', gateact='linear', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32)):
        
        if bn==True:
            lact = act
            act = 'linear'
            
        contents = self.conv1d_layer(x=x, ksize=cksize, fsize=fsize, name=name+'_cwc_contents',
                  bn=0, strides=cstrides, padding=padding,  L2 = L2,
                  act=act, trainable=trainable, reuse = reuse, W_init=W_init,b_init=b_init)
        weights = self.conv1d_layer(x=x, ksize=wksize, fsize=fsize, name=name+'_cwc_weights',
                  bn=0, strides=wstrides, padding=padding,  L2 = L2, 
                  act=gateact, trainable=trainable, reuse = reuse, W_init=W_init,b_init=b_init)
        weighted_content = contents*weights
        
        if bn==True:
            weighted_content = tf.layers.batch_normalization(weighted_content, training=(self.mode == tf.estimator.ModeKeys.TRAIN))
            weighted_content=self.activation(lact)(weighted_content)
            print(name+':add batchnorm')
        return weighted_content
    
    def cnn1d_inweighted_cnn1d_layer(self, x, cksize, wksize, fsize, name, bn=False, cstrides=1, wstrides=1, padding='same', L2 = 0, dp = 0,
              act='tanh', gateact='linear', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32)):
        
        if bn==True:
            lact = act
            act = 'linear'
            
        contents = self.conv1d_layer(x=x, ksize=cksize, fsize=fsize, name=name+'_cwc_contents',
                  bn=0, strides=cstrides, padding=padding,  L2 = L2,
                  act=act, trainable=trainable, reuse = reuse, W_init=W_init,b_init=b_init)
        weights = self.conv1d_layer(x=x, ksize=wksize, fsize=1, name=name+'_cwc_weights',
                  bn=0, strides=wstrides, padding=padding,  L2 = L2, 
                  act=gateact, trainable=trainable, reuse = reuse, W_init=W_init,b_init=b_init)
        weighted_content = contents*weights
        
        if bn==True:
            weighted_content = tf.layers.batch_normalization(weighted_content, training=(self.mode == tf.estimator.ModeKeys.TRAIN))
            weighted_content=self.activation(lact)(weighted_content)
            print(name+':add batchnorm')
        return weighted_content
    

    def cnn2d_weighted_cnn2d_layer(self, x, cksize, wksize, fsize, name, bn=False, cstrides=1, wstrides=1, padding='same', L2 = 0, dp = 0,
              act='relu', gateact='linear', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32)):
        
        if bn==True:
            lact = act
            act = 'linear'

        contents = self.conv2d_layer(x=x, ksize=cksize, fsize=fsize, name=name+'_cwc_contents',
                  bn=0, strides=cstrides, padding=padding,  L2 = L2,
                  act=act, trainable=trainable, reuse = reuse, W_init=W_init,b_init=b_init)
        weights = self.conv2d_layer(x=x, ksize=wksize, fsize=fsize, name=name+'_cwc_weights',
                  bn=0, strides=wstrides, padding=padding,  L2 = L2, 
                  act=gateact, trainable=trainable, reuse = reuse, W_init=W_init,b_init=b_init)
        weighted_content = contents*weights
        
        if bn==True:
            weighted_content = tf.layers.batch_normalization(weighted_content, training=(self.mode == tf.estimator.ModeKeys.TRAIN))
            weighted_content=self.activation(lact)(weighted_content)
            print(name+':add batchnorm')
        return weighted_content
    
    def rnn_weighted_cnn1d_layer(self, x, ksize, fsize, name, bn=False, strides=1, ctype = 'gru', bidirectional = 1, padding='same',  L2 = 0, dp = 0,
              act='relu', gateact='linear', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32)):
        
        if bn==True:
            lact = act
            act = 'linear'
        if bidirectional ==1:
            weights = self.birnn_layers(x, [fsize], name+'_rwc_weights', ctype=ctype, dp = dp, act=gateact)
            weights  = weights[:,:,:fsize]+weights[:,:,fsize:]
        elif bidirectional ==0:
            weights = self.rnn_layers(x, [fsize], name+'_rwc_weights', ctype=ctype, dp = dp, act=gateact)


        contents = self.conv1d_layer(x=x, ksize=ksize, fsize=fsize, name=name+'_rwc_contents', bn=0, strides=strides, padding=padding,  L2 = L2, dp = dp, act=act, trainable=trainable, reuse = reuse, W_init=W_init,b_init=b_init)
        
        weighted_content = contents*weights
        
        if bn==True:
            weighted_content = tf.layers.batch_normalization(weighted_content, training=(self.mode == tf.estimator.ModeKeys.TRAIN))
            weighted_content=self.activation(lact)(weighted_content)
            print(name+':add batchnorm')
        return weighted_content
    
    def rnn_indweighted_cnn1d_layer(self, x, ksize, fsize, name, bn=False, strides=1, ctype = 'gru', bidirectional = 1, padding='same',  L2 = 0, dp = 0,
              act='relu', gateact='linear', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32)):
        
        if bn==True:
            lact = act
            act = 'linear'
        if bidirectional ==1:
            weights = self.birnn_layers(x, [1], name+'_rwc_weights', ctype=ctype, dp = dp, act=gateact)
            weights  = weights[:,:,:1]+weights[:,:,1:]
        elif bidirectional ==0:
            weights = self.rnn_layers(x, [1], name+'_rwc_weights', ctype=ctype, dp = dp, act=gateact)


        contents = self.conv1d_layer(x=x, ksize=ksize, fsize=fsize, name=name+'_rwc_contents', bn=0, strides=strides, padding=padding,  L2 = L2, dp = dp, act=act, trainable=trainable, reuse = reuse, W_init=W_init,b_init=b_init)
        
        weighted_content = contents*weights
        
        if bn==True:
            weighted_content = tf.layers.batch_normalization(weighted_content, training=(self.mode == tf.estimator.ModeKeys.TRAIN))
            weighted_content=self.activation(lact)(weighted_content)
            print(name+':add batchnorm')
        return weighted_content
    
    def rnn_weighted_cnn2d_layer(self, x, ksize, fsize, input_size, name, bn=False, strides=1, ctype = 'gru',bidirectional = 1, 
              padding='same',L2 = 0, dp = 0,
              act='relu', gateact='linear', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32)):
        b,w,h,d = input_size
        wx = tf.reshape(x,[b,w,h*d])
        if bn==True:
            lact = act
            act = 'linear'
        if bidirectional ==1:
            weights = self.birnn_layers(wx, [h*fsize], name+'_rwc_weights', ctype=ctype, dp = dp, act=gateact)
            weights  = weights[:,:,:h*fsize]+weights[:,:,h*fsize:]
        elif bidirectional ==0:
            weights = self.rnn_layers(wx, [h*fsize], name+'_rwc_weights', ctype=ctype, dp = dp, act=gateact)

        weights = tf.reshape(weights,[b,w,h,fsize])
        
        contents = self.conv2d_layer(x=x, ksize=ksize, fsize=fsize, name=name+'_rwc_contents', bn=0, strides=strides, padding=padding, L2 = L2, dp = dp, act=act, trainable=trainable, reuse = reuse, W_init=W_init,b_init=b_init)
        
        weighted_content = contents*weights
        
        if bn==True:
            weighted_content = tf.layers.batch_normalization(weighted_content, training=(self.mode == tf.estimator.ModeKeys.TRAIN))
            weighted_content=self.activation(lact)(weighted_content)
            print(name+':add batchnorm')
        return weighted_content
    
    def rnn_indweighted_cnn2d_layer(self, x, ksize, fsize, input_size, name, bn=False, strides=1, ctype = 'gru', bidirectional = 1, 
              padding='same', L2 = 0, dp = 0,
              act='relu', gateact='linear', trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32)):
        b,w,h,d = input_size
        wx = tf.reshape(x,[b,w,h*d])
        if bn==True:
            lact = act
            act = 'linear'
        if bidirectional ==1:
            weights = self.birnn_layers(wx, [h], name+'_rwc_weights', ctype=ctype, dp = dp, act=gateact)
            weights  = weights[:,:,:h]+weights[:,:,h:]
        elif bidirectional ==0:
            weights = self.rnn_layers(wx, [h], name+'_rwc_weights', ctype=ctype, dp = dp, act=gateact)

        weights = tf.reshape(weights,[b,w,h,1])
        
        contents = self.conv2d_layer(x=x, ksize=ksize, fsize=fsize, name=name+'_rwc_contents', bn=0, strides=strides, padding=padding, L2 = L2, dp = dp, act=act, trainable=trainable, reuse = reuse, W_init=W_init,b_init=b_init)
        
        weighted_content = contents*weights
        
        if bn==True:
            weighted_content = tf.layers.batch_normalization(weighted_content, training=(self.mode == tf.estimator.ModeKeys.TRAIN))
            weighted_content=self.activation(lact)(weighted_content)
            print(name+':add batchnorm')
        return weighted_content
