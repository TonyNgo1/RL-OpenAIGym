import numpy as np
import random

class experienceBuffer():
    #Initialize an empty buffer of buffer_size
    def __init__(self, buffer_size = 2000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    #Add experience to the buffer, and clear old experiences of full
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    #Take a random sample of the experiences to work with
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])