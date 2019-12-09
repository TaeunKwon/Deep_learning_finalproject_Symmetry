from sklearn.manifold import TSNE
import os
import sys
import numpy as np
import tensorflow as tf
from autoencoder import AutoEncoder
from Cluster import clustering
import time
import preprocess as preprocess
import visualization
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


tf.enable_eager_execution()

def test(model, test_data, encoded):
    #encoded = model.call(test_data)
    #loss = model.loss_function(encoded, test_data)
    BSZ = model.batch_size
    curr_loss = 0
    step = 0
    for start, end in zip(range(0, len(test_data) - BSZ, BSZ), range(BSZ, len(test_data), BSZ)):
        encoded[start:end,:] = model.encoder.call(test_data[start:end]).numpy()
        
        if step % 10 == 0:
            print('%dth batches, \tAvg. loss: %.3f' % (step, loss/BSZ))


def cluster_sklearn():    
    pulse_data, label, test_data, test_label, evt_ind, ch_ind = preprocess.get_data("../testData11_14bit_100mV.npy", 5500, 5000)
  
    model = AutoEncoder()
    checkpoint_dir = './checkpoint'
    checkpoint = tf.train.Checkpoint(model = model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    
    encoded = np.zeros([len(test_data),100])
    test(model, test_data, encoded)
    clustering = DBSCAN().fit(encoded)
    
    visualization.feature_v_proj_v2(encoded, test_label, clustering.labels_)
    
if __name__ == '__main__':
	cluster_sklearn()
