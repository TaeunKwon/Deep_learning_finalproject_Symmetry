import os
import sys
import numpy as np
import tensorflow as tf

class cluster(tf.keras.layers.Layer):
    def __init__(self, n_clusters, weights = None, alpha = 1.0):
        super(cluster, self).__init__()
        
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.centroid_weights = weights
        
    def build(self, input_shape):
        assert len(input_shape) == 2
        
        input_dim = input_shape[1]
        self.clusters = self.add_weight(shape = (self.n_clusters, input_dim), initializer='random_normal', trainable = True)
        if self.centroid_weights is not None:
            self.set_weights(self.centroid_weights)
            del self.centroid_weights
        self.built = True
        
    #@tf.function
    def call(self, inputs):
        q = 1.0 / (1.0 + (tf.math.reduce_sum(tf.math.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tf.transpose(tf.transpose(q) / tf.math.reduce_sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q
    
class clustering(tf.keras.Model):
    def __init__(self, encoder):
        super(clustering, self).__init__()
        self.batch_size = 2000
        self.learning_rate = 0.005
        self.n_clusters = 3
        
        self.encoder = encoder
        self.cluster = cluster(n_clusters = self.n_clusters)
        self.cluster.build([None, 100])
        
        self.optimizer =  tf.keras.optimizers.Adam(self.learning_rate)
        
    #@tf.function
    def call(self, pulses):
        #print('autoencoder call function')
        return self.cluster.call(self.encoder.call(pulses))
    
    def target_distribution(self, q):
        weight = q**2 /tf.math.reduce_sum(q, axis = 0)
        return tf.transpose((tf.transpose(weight) / tf.math.reduce_sum(weight, axis = 1)))
    
    #@tf.function
    def loss_function(self, q, p):
      #Q = tf.distributions.Categorical(probs = q)
      #P = tf.distributions.Categorical(probs = q)
      #return tf.distributions.kl_divergence(Q,P)
      return tf.reduce_sum(p*tf.math.log(p/q))#*tf.math.square(tf.dtypes.cast(tf.reduce_sum(tf.math.argmin(q, axis = 1)), dtype = tf.float32) - q.shape[0]/2)
