import os
import sys
import numpy as np
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        
        #Hyperparameters
        
        self.filter_size1 = 10
        self.filter_size2 = 10
        self.filter_size3 = 4
        
        self.kernel_size1 = 10
        self.kernel_size2 = 10
        self.kernel_size3 = 10
        
        self.stride1 = 2
        self.stride2 = 2
        self.stride3 = 3
        
        self.pool_size1 = 3
        self.pool_size2 = 3
        
        self.Dense_size = 200
        
        #Layers
        self.encoder_conv1 = tf.keras.layers.Conv1D(filters = self.filter_size1, kernel_size = self.kernel_size1, strides = self.stride1,
                                                    padding = 'same', activation = tf.keras.layers.LeakyReLU(alpha = 0.2), kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.encoder_maxpool1 = tf.keras.layers.MaxPool1D(pool_size = self.pool_size1, padding = 'same')
        self.encoder_conv2 = tf.keras.layers.Conv1D(filters = self.filter_size2, kernel_size = self.kernel_size2, strides = self.stride2,
                                                    padding = 'same', activation = tf.keras.layers.LeakyReLU(alpha = 0.2), kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.encoder_maxpool2 = tf.keras.layers.MaxPool1D(pool_size = self.pool_size2, padding = 'same')
        self.encoder_conv3 = tf.keras.layers.Conv1D(filters = self.filter_size3, kernel_size = self.kernel_size3, strides = self.stride3,
                                                    padding = 'same', activation = tf.keras.layers.LeakyReLU(alpha = 0.2), kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        #self.Dense1 = tf.keras.layers.Dense(self.Dense_size, activation = tf.keras.layers.ReLU(),dtype = tf.float32)
        
    def call(self, pulses):
        
        output = self.encoder_conv1(pulses)
        #print("Encoder after conv1: ",output.get_shape())
        output = self.encoder_maxpool1(output)
        #print("Encoder after maxpool1: ",output.get_shape())
        output = self.encoder_conv2(output)    
        #print("Encoder after conv2: ",output.get_shape())
        output = self.encoder_maxpool2(output)
        #print("Encoder after maxpool2: ",output.get_shape())
        #output = self.Dense1(tf.reshape(output, [-1, tf.shape(output)[1]*tf.shape(output)[2]]))
        output = self.encoder_conv3(output)
        #print("Encoder output: ",output.get_shape())
        return output

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        
        #Hyperparameters
        self.Dense_size = 225*4
        
        self.filter_size1 = 10
        self.filter_size2 = 10
        self.filter_size3 = 4
        
        self.kernel_size1 = 10
        self.kernel_size2 = 10
        self.kernel_size3 = 10
        
        self.stride1 = 3
        self.stride2 = 2
        self.stride3 = 2
        
        self.pool_size1 = 3
        self.pool_size2 = 3
                
        #Layers
        self.deconv1_W = tf.Variable(tf.random.normal([self.kernel_size1, self.filter_size1, 4], stddev = 0.1))
        self.deconv1_b = tf.Variable(tf.random.normal([self.filter_size1,], stddev = 0.1))
        
        self.deconv2_W = tf.Variable(tf.random.normal([self.kernel_size2, self.filter_size2, self.filter_size1], stddev = 0.1))
        self.deconv2_b = tf.Variable(tf.random.normal([self.filter_size2,], stddev = 0.1))
        
        self.deconv3_W = tf.Variable(tf.random.normal([self.kernel_size3, self.filter_size3, self.filter_size2], stddev = 0.1))
        self.deconv3_b = tf.Variable(tf.random.normal([self.filter_size3,], stddev = 0.1))
        
        self.Dense1 = tf.keras.layers.Dense(self.Dense_size, dtype = tf.float32)
        self.decoder_upsample1 = tf.keras.layers.UpSampling1D(size = self.pool_size1)
        self.decoder_upsample2 = tf.keras.layers.UpSampling1D(size = self.pool_size2)

    def call(self, encoder_output):
        batchSz = tf.shape(encoder_output)[0]
        #output = tf.reshape(self.Dense1(encoder_output), (-1,225,4))
        #print("Decoder Input: ",encoder_output.get_shape())
        output = tf.nn.conv1d_transpose(encoder_output, self.deconv1_W, output_shape = [batchSz, 300, self.filter_size1], strides = self.stride1, padding = 'SAME')
        output = tf.nn.leaky_relu(tf.add(output, self.deconv1_b))
        #print("Decoder after deconv1: ",output.get_shape())
        output = self.decoder_upsample1(output)
        #print("Decoder after upsample1: ",output.get_shape())
        output = tf.nn.conv1d_transpose(output, self.deconv2_W, output_shape = [batchSz, 1800, self.filter_size2], strides = self.stride2, padding = 'SAME')
        output = tf.nn.leaky_relu(tf.add(output, self.deconv2_b))
        #print("Decoder after deconv2: ",output.get_shape())
        output = self.decoder_upsample2(output)
        #print("Decoder after upsample2: ",output.get_shape())
        output = tf.nn.conv1d_transpose(output, self.deconv3_W, output_shape = [batchSz, 10800, self.filter_size3], strides = self.stride3, padding = 'SAME')
        output = tf.nn.leaky_relu(tf.add(output, self.deconv3_b))
        #print("Decoder output: ",output.get_shape())
        return output

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.batch_size = 100
        self.learning_rate = 0.01
        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
        #self.optimizer =  tf.keras.optimizers.Adam(self.learning_rate)
    def call(self, pulses):
        return self.decoder(self.encoder(pulses))
    
    def loss_function(self, encoded, originals):
      encoded = tf.dtypes.cast(encoded, tf.float32)
      originals = tf.dtypes.cast(originals, tf.float32)
      return tf.reduce_sum((originals - encoded)*(originals - encoded))
