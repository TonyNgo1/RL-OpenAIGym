import tensorflow as tf

class Qnetwork(object):
    def __init__(self, inputSize, num_actions):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through three convolution-pooling layers.
        self.scalarInput =  tf.placeholder(shape=[None,inputSize],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,4])
        self.imageIn = tf.image.resize_images(self.imageIn, [80,80]) 
        
        self.conv1=self.conv2d(name='conv1', inputs=self.imageIn, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='SAME', activation_fn=tf.nn.elu)
        self.pool1 = self.max_pool2d(name='pool1', inputs=self.conv1, kernel_size=[2,2], stride=[2,2], padding='SAME')
        self.conv2=self.conv2d('conv2', inputs=self.pool1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='SAME', activation_fn=tf.nn.elu)
        self.pool2 = self.max_pool2d(name='pool2', inputs=self.conv2, kernel_size=[2,2], stride=[2,2], padding='SAME')
        self.conv3=self.conv2d('conv3', inputs=self.pool2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='SAME', activation_fn=tf.nn.elu)
        self.pool3 = self.max_pool2d(name='pool3', inputs=self.conv3, kernel_size=[2,2], stride=[2,2], padding='SAME')
        
        #Send the output of the final pooling layer and send it into a dense layer
        self.rs = tf.reshape(self.pool3, shape=[-1,256])
        self.fc1 = self.fully_connected('fc1', self.rs, 256, activation_fn=tf.nn.elu)
        
        #We take the output from the dense and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.fc1,2,1)
        self.streamA = tf.reshape(self.streamAC, shape=[-1,128])
        self.streamV = tf.reshape(self.streamVC, shape=[-1,128])
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([128,num_actions]))
        self.VW = tf.Variable(xavier_init([128,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,num_actions,dtype=tf.float32)
        
        #Q value of action we chose
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
        self.gradients, self.variables = zip(*self.trainer.compute_gradients(self.loss))
        #self.updateModel = self.trainer.minimize(self.loss)
        self.grads_clipped, _ = tf.clip_by_global_norm(self.gradients,0.1)
        self.updateModel = self.trainer.apply_gradients(zip(grads_clipped,variables))
        
    #Function helper to define a 2D convolution layer
    def conv2d(self, name, inputs, num_outputs, kernel_size, stride, padding, activation_fn):
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal([kernel_size[0],kernel_size[1],inputs.get_shape().as_list()[-1],num_outputs], dtype=tf.float32, stddev=0.02))
            conv = tf.nn.conv2d(inputs, kernel, [1,stride[0],stride[1],1], padding=padding)
            biases = tf.Variable(tf.constant(0.0, shape=[num_outputs], dtype=tf.float32), trainable=True)
            conv_bias = tf.nn.bias_add(conv, biases)
            if (activation_fn):
                return activation_fn(conv_bias)
            return conv_bias
    
    #Function helper to define a 2D max pooling layer
    def max_pool2d(self, name, inputs, kernel_size, stride, padding):
        with tf.name_scope(name) as scope:
            pool = tf.nn.max_pool(inputs, [1,kernel_size[0],kernel_size[1],1], [1,stride[0],stride[1],1], padding=padding)
            return pool
       
    #Function helper to define a fully_connected (dense) layer
    def fully_connected(self, name, inputs, num_outputs, activation_fn):
        with tf.name_scope(name) as scope:
            weights = tf.Variable(tf.truncated_normal([inputs.get_shape().as_list()[-1],num_outputs]))
            biases = tf.Variable(tf.constant(0.0, shape=[num_outputs], dtype=tf.float32), trainable=True)
            fc = tf.matmul(inputs, weights) + biases
            if (activation_fn):
                return activation_fn(fc)
            return fc