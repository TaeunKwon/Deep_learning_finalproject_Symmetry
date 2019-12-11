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
        self.clusters = self.add_weight(shape = (self.n_clusters, input_dim), initializer='random_normal', trainable = True, name = "clusters")
        if self.centroid_weights is not None:
            self.set_weights(self.centroid_weights)
            del self.centroid_weights
        self.built = True
        
    #@tf.function
    def call(self, inputs):
        
        q = 1.0 / (1.0 + ( self.siml(inputs)/ self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tf.transpose(tf.transpose(q) / tf.math.reduce_sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q
    
    def siml(self, inputs):
        complexity_input = tf.math.sqrt(tf.reduce_sum(tf.square(inputs[:, 1:] - inputs[:, :-1]), axis = 1))
        complexity_cent = tf.math.sqrt(tf.reduce_sum(tf.square(self.clusters[:, 1:] - self.clusters[:, :-1]), axis = 1))   
        complexity_factor = tf.zeros((len(inputs), self.n_clusters))
        C_input, C_cent = tf.meshgrid(complexity_cent, complexity_input)
        complexity_factor = tf.keras.backend.maximum(C_input, C_cent)/tf.keras.backend.minimum(C_input, C_cent)
        
        euclidean = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2))
        
        return euclidean*complexity_factor
        
    
class clustering(tf.keras.Model):
    def __init__(self, encoder):
        super(clustering, self).__init__()
        self.batch_size = 1000
        self.learning_rate = 1e-3#0.00001
        self.n_clusters = 3
        
        self.encoder = encoder
        
        self.cluster = cluster(n_clusters = self.n_clusters)
        self.cluster.build([None, 10])
        
        self.optimizer =  tf.keras.optimizers.Adam(self.learning_rate)
        
    #@tf.function
    def call(self, pulses):
        #print('autoencoder call function')
        return self.cluster.call(self.encoder(pulses))
    
    def target_distribution(self, q):
        weight = q**2 /tf.math.reduce_sum(q, axis = 0)
        return tf.transpose((tf.transpose(weight) / tf.math.reduce_sum(weight, axis = 1)))
    
    #@tf.function
    def loss_function(self, q, p):
      #Q = tf.distributions.Categorical(probs = q)
      #P = tf.distributions.Categorical(probs = q)
      #return tf.distributions.kl_divergence(Q,P)
      
      return tf.keras.losses.KLDivergence()(p, q)*tf.square((tf.cast(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(q, axis = 1), tf.cast(tf.zeros(len(q)), dtype = tf.int64)), dtype = tf.int32)), dtype = tf.float32) - len(q)/2.))
