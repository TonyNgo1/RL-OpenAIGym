import sys
import gym
import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import os
#from helper import *
from skimage.color import rgb2gray
from random import choice
from time import sleep
from time import time
import collections
import warnings
# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
    #s = frame[10:-10,30:-30]
    #s = scipy.misc.imresize(s,[84,84])
    #s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    #return s
    s = rgb2gray(frame)
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)])
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class AC_Network():
    def __init__(self,s_size,frames_stacked,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,frames_stacked,s_size],dtype=tf.float32)
            #Reshape to states, with stacked frames as channels
            #And then transpose from NCHW to NHWC format
            self.imageIn = tf.transpose(tf.reshape(self.inputs,shape=[-1,4,84,84]),[0,2,3,1])
            
            '''self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=16,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=32,
                kernel_size=[4,4],stride=[2,2],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)'''
            
            
            #print ("IMAGEIN " + str(self.imageIn))
            self.conv1 = slim.conv2d( \
                inputs=self.imageIn,num_outputs=32,kernel_size=[5,5],stride=[1,1],padding='SAME', activation_fn=tf.nn.elu, biases_initializer=None)
            #print ("CONV 1 OUTS " + str(self.conv1))
            self.pool1 = slim.max_pool2d(self.conv1, [2,2], [2,2], padding='SAME')
            #print ("POOL 1 OUTS " + str(self.pool1))
            self.conv2 = slim.conv2d( \
                inputs=self.pool1,num_outputs=32,kernel_size=[5,5],stride=[1,1],padding='SAME', activation_fn=tf.nn.elu,biases_initializer=None)
            #print ("CONV 2 OUTS " + str(self.conv2))
            self.pool2 = slim.max_pool2d(self.conv2, [2,2], [2,2], padding='SAME')
            #print ("POOL 2 OUTS " + str(self.pool2))
            self.conv3 = slim.conv2d( \
                inputs=self.pool2,num_outputs=64,kernel_size=[4,4],stride=[1,1],padding='SAME',activation_fn=tf.nn.elu, biases_initializer=None)
            #print ("CONV 3 OUTS " + str(self.conv3))
            self.pool3 = slim.max_pool2d(self.conv3, [2,2], [2,2], padding='SAME')
            #print ("POOL 3 OUTS " + str(self.pool3))
            self.conv4 = slim.conv2d( \
                inputs=self.pool3,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID',activation_fn=tf.nn.relu, biases_initializer=None)
            #print ("CONV 4 OUTS " + str(self.conv4))     
            self.pool4 = slim.max_pool2d(self.conv4, [2,2], [2,2], padding='VALID')
            #print ("POOL 4 OUTS " + str(self.pool4)) 
            self.rs = tf.reshape(self.pool4, shape=[-1,1024])
            #print ("RESHAPE OUTS " + str(self.rs))
            self.fc1 = slim.fully_connected(self.rs, 1024, activation_fn=tf.nn.elu)
            #print ("FC1 OUTS " + str(self.fc1))
            
            
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(self.fc1,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(self.fc1,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.log_policy = tf.log(tf.clip_by_value(self.policy, 1e-20, 100.0)) # avoid NaN with clipping when value in policy becomes zero
                self.entropy = - tf.reduce_sum(self.policy * self.log_policy)
                self.log_responsible_outputs=tf.log(tf.clip_by_value(self.responsible_outputs, 1e-20, 100.0))
                self.policy_loss = -tf.reduce_sum(self.log_responsible_outputs*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,0.1)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

class Worker():
    #workers.append(Worker(DoomGame(),i,s_size,a_size,trainer,model_path,global_episodes))
    def __init__(self,game,name,s_size,frames_stacked,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,frames_stacked,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        self.actions = self.actions = np.identity(a_size,dtype=bool).tolist()
        
        self.env = game
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        #print ("RS: " + str(np.shape(np.vstack(rollout[:,0]))))
        observations = np.reshape(np.vstack(rollout[:,0]), [-1,4,7056])
        #print ("OBS: " + str(np.shape(observations)))
        #print ("OBS1: " + str(np.shape(observations[0])))
        #print ("OBS2: " + str(np.shape(observations[0][0])))
        #print (str(observations[0][0][7055]))
        actions = rollout[:,1]
        rewards = rollout[:,2]
        #next_observations = rollout[:,3]
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            #self.local_AC.inputs:np.vstack(observations),
            self.local_AC.inputs:observations,
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                #episode_buffer = []
                episode_buffer = collections.deque(maxlen=100)
                state_buffer = collections.deque(maxlen=4)
                episode_values = []
                #episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                s = self.env.reset()
                #episode_frames.append(s)
                s = process_frame(s)
                for i in range(4):
                    state_buffer.append(s)
                #while self.env.is_episode_finished() == False:
                #print (str(np.shape(state_buffer)))
                while d == False:
                    #Take an action using probabilities from policy network output.
                    a_dist,v= sess.run([self.local_AC.policy,self.local_AC.value], 
                        feed_dict={self.local_AC.inputs:[state_buffer]})
                    #print("STB: " + str(np.shape(np.expand_dims(state_buffer,0))))
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error')
                        try:
                            a = np.random.choice(a_dist[0],p=a_dist[0])
                        except Warning as e:
                            print (str(e))
                            exit()
                    a = np.argmax(a_dist == a)

                    #r = self.env.make_action(self.actions[a]) / 100.0
                    #d = self.env.is_episode_finished()
                    s1,r_,d,_ = self.env.step(self.actions[a])
                    r = max(-1,min(1,r_))
                    if self.name == 'worker_0' and episode_count % 10 == 0:
                        self.env.render()
                    '''if d == False:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                    else:
                        s1 = s'''
                    #episode_frames.append(s1)
                    s1 = process_frame(s1)
                    buf_old=np.array(state_buffer)
                    state_buffer.append(s1)
                    episode_buffer.append([buf_old,a,r,np.array(state_buffer),d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r_
                    #s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if episode_step_count%100==0 and d != True and episode_step_count != max_episode_length - 1:
                    #if len(episode_buffer) == 30 and d != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        #batch = np.reshape(np.array(random.sample(episode_buffer,100)),[100,6])
                        #batch = np.reshape(episode_buffer,[100,6])
                        #print("BTS: " + str(batch))
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[buf_old]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        #episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
                                
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    '''if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.02
                        images = np.array(episode_frames)
                        make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)'''
                    if episode_count % 100 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    print ("WORKER: {} EPISODE: {} MEAN REWARD: {} MEAN LENGTH: {} MEAN VALUE: {}".format(self.name, episode_count, mean_reward, mean_length, mean_value))
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                
max_episode_length = 6000
gamma = 0.99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
frames_stacked = 4 #Amount of frames stacked
#a_size = 3 # Agent can move Left, Right, or Fire
a_size = gym.make('Breakout-v0').action_space.n
load_model = False
model_path = './a3cbreakout'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
'''#Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')'''

#with tf.device("/cpu:0"): 
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
#trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
trainer = tf.train.RMSPropOptimizer(learning_rate=0.00025,decay=0.99,use_locking=True)
master_network = AC_Network(s_size,frames_stacked,a_size,'global',None) # Generate global network
num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
workers = []
# Create worker classes
for i in range(num_workers):
    workers.append(Worker(gym.make('Breakout-v0'),i,s_size,frames_stacked,a_size,trainer,model_path,global_episodes))
saver = tf.train.Saver(max_to_keep=5)
#end block

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)