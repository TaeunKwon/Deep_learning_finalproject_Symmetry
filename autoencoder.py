import os
import sys
import numpy as np
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        
        #Hyperparameters
        self.filter_size1 = 10
        self.filter_size2 = 4
        
        self.kernel_size1 = 10
        self.kernel_size2 = 10
        
        self.stride1 = 2
        self.stride2 = 3
        
        self.pool_size1 = 2
        
        self.Dense_size = 10
        
        #Layers
        self.encoder_conv1 = tf.keras.layers.Conv1D(filters = self.filter_size1, kernel_size = self.kernel_size1, strides = self.stride1,
                                                    padding = 'same', activation = tf.keras.layers.LeakyReLU(alpha = 0.2), kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.encoder_maxpool1 = tf.keras.layers.MaxPool1D(pool_size = self.pool_size1, padding = 'same')
        self.encoder_conv2 = tf.keras.layers.Conv1D(filters = self.filter_siz2, kernel_size = self.kernel_size2, strides = self.stride2,
                                                    padding = 'same', activation = tf.keras.layers.LeakyReLU(alpha = 0.2), kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.Dense1 = tf.keras.layers.Dense(self.Dense_size, dtype = tf.float32)
        
    @tf.function
    def call(self, pulses):
        output = self.encoder_conv1(pulses)
        output = self.encoder_maxpool1(output)
        output = self.encoder_conv2(output)
        output = self.Dense1(output)
        return output

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        
        #Hyperparameters
        self.Dense_size = 225
        
        self.filter_size1 = 10
        self.filter_size2 = 4
        
        self.kernel_size1 = 10
        self.kernel_size2 = 10
        
        self.stride1 = 3
        self.stride2 = 2
        
        self.pool_size1 = 2
                
        #Layers
        self.Dense1 = tf.keras.layers.Dense(self.Dense_size, dtype = tf.float32)
        self.decoder_deconv1 = tf.keras.layers.Conv1DTranspose(filters = self.filter_size1, kernel_size = self.kernel_size1, strides = self.stride1,
                                                    padding = 'same', activation = tf.keras.layers.LeakyReLU(alpha = 0.2), kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.decoder_upsample1 = tf.keras.layers.UpSampling1D(size = self.pool_size1)
        self.decoder_deconv2 = tf.keras.layers.Conv1DTranspose(filters = self.filter_siz2, kernel_size = self.kernel_size2, strides = self.stride2,
                                                    padding = 'same', activation = tf.keras.layers.LeakyReLU(alpha = 0.2), kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        
    @tf.function
    def call(self, encoder_output):
        output = self.Dense1(encoder_output)
        output = self.decoder_deconv1(output)
        output = self.decoder_upsample1(output)
        output = self.decoder_deconv2(output)
        return output

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    @tf.function
    def call(self, pulses):
       return self.decoder(self.encoder(pulses))
    
    @tf.function
    def loss_function(self, encoded, originals):
      encoded = tf.dtypes.cast(encoded, tf.float32)
      originals = tf.dtypes.cast(originals, tf.float32)
      return tf.reduce_sum((originals - encoded)*(originals - encoded))
