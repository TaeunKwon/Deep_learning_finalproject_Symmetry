import os
import sys
import numpy as np
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        
        #Hyperparameters
        
        self.filter_size1 = 2
        
        self.kernel_size1 = 10
        
        self.stride1 = 1
        
        self.pool_size1 = 2
        
        self.Dense_size1 = 800
        self.Dense_size2 = 400
        self.Dense_size3 = 100
        
        #Layers
        self.encoder_conv1 = tf.keras.layers.Conv1D(filters = self.filter_size1, kernel_size = self.kernel_size1, strides = self.stride1,
                                                    padding = 'same', activation = tf.keras.layers.LeakyReLU(alpha = 0.2), kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.encoder_maxpool1 = tf.keras.layers.MaxPool1D(pool_size = self.pool_size1, padding = 'same')
        self.Dense1 = tf.keras.layers.Dense(self.Dense_size1, activation = tf.keras.layers.LeakyReLU(),dtype = tf.float32)
        self.Dense2 = tf.keras.layers.Dense(self.Dense_size2, activation = tf.keras.layers.LeakyReLU(),dtype = tf.float32)
        self.Dense3 = tf.keras.layers.Dense(self.Dense_size3, activation = tf.keras.layers.LeakyReLU(),dtype = tf.float32)
        
    #@tf.function
    def call(self, pulses):
        output = tf.keras.layers.BatchNormalization()(self.encoder_conv1(pulses))
        output = tf.keras.layers.BatchNormalization()(self.encoder_maxpool1(output))
        output = tf.keras.layers.BatchNormalization()(self.Dense1(tf.reshape(output, [-1, tf.shape(output)[1]*tf.shape(output)[2]])))
        output = tf.keras.layers.BatchNormalization()(self.Dense2(output))
        output = tf.keras.layers.BatchNormalization()(self.Dense3(output))
        return output

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        
        #Hyperparameters
        self.Dense_size1 = 400
        self.Dense_size2 = 800
        self.Dense_size3 = 1300# =numSamples / pool_size1 * filter_size1
        
        self.filter_size3 = 1
        
        self.kernel_size3 = 10
        
        self.stride3 = 1
        
        self.pool_size1 = 2
                
        #Layers
        
        self.deconv3_W = tf.random.normal([self.kernel_size3, self.filter_size3, 2], stddev = 0.1)
        self.deconv3_b = tf.random.normal([self.filter_size3,], stddev = 0.1)
        
        self.Dense1 = tf.keras.layers.Dense(self.Dense_size1, activation = tf.keras.layers.LeakyReLU(), dtype = tf.float32)
        self.Dense2 = tf.keras.layers.Dense(self.Dense_size2, activation = tf.keras.layers.LeakyReLU(), dtype = tf.float32)
        self.Dense3 = tf.keras.layers.Dense(self.Dense_size3, activation = tf.keras.layers.LeakyReLU(), dtype = tf.float32)
        self.decoder_upsample1 = tf.keras.layers.UpSampling1D(size = self.pool_size1)
        
    #@tf.function
    def call(self, encoder_output):
        batchSz = tf.shape(encoder_output)[0]
        
        output = tf.keras.layers.BatchNormalization()(self.Dense1(encoder_output))
        output = tf.keras.layers.BatchNormalization()(self.Dense2(output))
        output = tf.keras.layers.BatchNormalization()(tf.reshape(self.Dense3(output), (-1,650,2)))
        output = tf.keras.layers.BatchNormalization()(tf.nn.leaky_relu(self.decoder_upsample1(output)))
        output = tf.nn.conv1d_transpose(output, self.deconv3_W, output_shape = [batchSz, int(self.stride3*output.get_shape()[1]), self.filter_size3], strides = self.stride3, padding = 'SAME')
        output = tf.keras.layers.BatchNormalization()(tf.nn.leaky_relu(tf.add(output, self.deconv3_b)))
        return output

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.batch_size = 100
        self.learning_rate = 1e-3
        self.learning_rate2 = 1e-3
        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
        self.optimizer =  tf.keras.optimizers.Adam(self.learning_rate)
        self.optimizer2 =  tf.keras.optimizers.Adam(self.learning_rate2)
        
    #@tf.function
    def call(self, pulses):
        return self.decoder.call(self.encoder.call(pulses))
    
    #@tf.function
    def loss_function(self, encoded, originals):
      encoded = tf.dtypes.cast(encoded, tf.float32)
      originals = tf.dtypes.cast(originals, tf.float32)
      return tf.reduce_sum((originals - encoded)*(originals - encoded))
