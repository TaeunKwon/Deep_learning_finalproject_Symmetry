# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 03:02:15 2019

@author: tu889
"""

import os
import sys
import numpy as np
import tensorflow as tf

class Cluster(tf.keras.Model):
    def __init__(self):
        super(Cluster, self).__init__()
        self.batch_size = 100
        self.learning_rate = 0.001
        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
        self.optimizer =  tf.keras.optimizers.Adam(self.learning_rate)
        
    #@tf.function
    def call(self, pulses):
        #print('autoencoder call function')
        return self.decoder.call(self.encoder.call(pulses))
    
    #@tf.function
    def loss_function(self, encoded, originals):
      encoded = tf.dtypes.cast(encoded, tf.float32)
      originals = tf.dtypes.cast(originals, tf.float32)
      return tf.reduce_sum((originals - encoded)*(originals - encoded))
