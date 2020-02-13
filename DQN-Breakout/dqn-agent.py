from __future__ import division
import gym
import numpy as np
import random
import tensorflow as tf
import scipy.misc
import os
from skimage.color import rgb2gray

from network import Qnetwork
from experience_replay import experienceBuffer

#Convert the raw image from RGB to grayscale, and then resize to our input_h*input_w
def processState(frame):
    s = rgb2gray(frame)
    s = scipy.misc.imresize(s,[input_h,input_w])
    s = np.reshape(s,[np.prod(s.shape),1])
    return s

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
        

batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 500. #How many eps of training to reduce startE to endE.
num_episodes = 100000 #How many episodes of game environment to train network with.
pre_train_eps=300 #How many episodes of random action before training begins
load_model = False #Whether to load a saved model.
path = "./dqn-models" #The path to save our model to.
tau = 0.001 #Rate to update target network toward primary network
input_h=84 #Height of the image when sending to our network
input_w=84 #Width of the image when sending to our network
frames_stacked=4 #The amount of frames to stack
input_size=input_h*input_w*frames_stacked #The total size of each state, as HxWxC

env = gym.make('Breakout-v0')

tf.reset_default_graph()
mainQN = Qnetwork(input_size, env.action_space.n)
targetQN = Qnetwork(input_size, env.action_space.n)
replayBuffer = experienceBuffer()

init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)

#Set the rate of random action decrease. 
e = startE
epDrop = (startE - endE)/annealing_steps

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

sess = tf.Session()
sess.run(init)

if load_model == True:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    
for i in range(num_episodes):
    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0

    episodeBuffer = experienceBuffer()
    
    #Reset environment and get first new observation
    s = env.reset()
    s = processState(s)
    state_stack=np.empty([input_h*input_w, 1])
    for k in range(0, frames_stacked):
        state_stack=np.append(state_stack,s,axis=1)
    while(np.shape(state_stack)[1]>4):
        state_stack=np.delete(state_stack, 0, axis=1)
        
    state_stack1=np.copy(state_stack)
    d = False
    rAll = 0
    j = 0
    
    if i >= pre_train_eps:
        if e > endE:
                e -= epDrop 
    while True:
        j+=1
        #Choose an action by greedily (with e chance of random action) from the Q-network
        if np.random.rand(1) < e or i < pre_train_eps:
            a = np.random.randint(0,env.action_space.n)
        else:
            a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[np.reshape(state_stack, [input_size])]})[0]
        
        s1,r_,d,_ = env.step(a)
        r = max(-1,min(1,r_))        
        s1 = processState(s1)
        state_stack1=np.append(state_stack1,s1,axis=1)
        while(np.shape(state_stack1)[1]>4):
            state_stack1=np.delete(state_stack1, 0, axis=1)
        total_steps += 1
        
        episodeBuffer.add(np.reshape(np.array([np.reshape(state_stack, [input_size]),a,r,np.reshape(state_stack1, [input_size]),d]),[1,5])) #Save the experience to our episode buffer.
        
        if i>=pre_train_eps and i % 20 == 0:
            env.render()
        
        if i >= pre_train_eps:
            if total_steps % (update_freq) == 0:
                trainBatch = replayBuffer.sample(batch_size) #Get a random batch of experiences.
                #Below we perform the Double-DQN update to the target Q-values
                Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                end_multiplier = -(trainBatch[:,4] - 1)
                doubleQ = Q2[range(batch_size),Q1]
                targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                #Update the network with our target values.
                _ = sess.run(mainQN.updateModel, \
                    feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                
                updateTarget(targetOps,sess) #Update the target network toward the primary network.
        rAll += r_
        state_stack = np.copy(state_stack1)
        
        if d == True:
            break
    
    replayBuffer.add(episodeBuffer.buffer)
    jList.append(j)
    rList.append(rAll)
    #Periodically save the model. 
    if i % 100 == 0:
        saver.save(sess,path+'/model-'+str(i)+'.ckpt')
        print("Saved Model")
    if i % 5 == 0:
        print(i,np.mean(rList[-10:]), e)
saver.save(sess,path+'/model-'+str(i)+'.ckpt')
