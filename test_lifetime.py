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
from lifetime import lifetime
from sklearn.manifold import TSNE

tf.enable_eager_execution()

def test(model, test_data, encoded):
    #encoded = model.call(test_data)
    #loss = model.loss_function(encoded, test_data)
    BSZ = model.batch_size
    step = 0
    for start, end in zip(range(0, len(test_data) - BSZ, BSZ), range(BSZ, len(test_data), BSZ)):
        encoded[start:end,:] = model.encoder.call(test_data[start:end]).numpy()
        step+=1
        if step % 10 == 0:
            print('%dth batches done.' % (step))
            
def restore_cluster():
    pulse_data, label, test_data, test_label, evt_ind, ch_ind = preprocess.get_data("../testData11_14bit_100mV.npy", 50010, 50000)

    model = AutoEncoder()
    #checkpoint_dir = './checkpoint'
    #checkpoint = tf.train.Checkpoint(model = model)
    #manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    #checkpoint.restore(manager.latest_checkpoint)
    
    model_cluster = clustering(model)
    checkpoint_dir_cluster = './checkpoint_cluster'
    checkpoint_cluster = tf.train.Checkpoint(model = model_cluster)
    manager_cluster = tf.train.CheckpointManager(checkpoint_cluster, checkpoint_dir_cluster, max_to_keep=3)
    checkpoint_cluster.restore(manager_cluster.latest_checkpoint)

    encoded = np.zeros([len(test_data),100])
    test(model_cluster.autoencoder, test_data, encoded)
    #np.save("./results/temp_encoded_50000evts.npy",encoded)
    
    #transformed = np.load("./results/transformed.npy")
    transformed = TSNE(n_components=2).fit_transform(encoded)
    #np.save("./results/temp_transformed.npy",transformed)
    
    plt.scatter(transformed[:, 0], transformed[:, 1], c=test_label, s = 4)
    plt.colorbar()
    plt.show()
            
def muon_lifetime():
    
    pulse_data, label, test_data, test_label, evt_ind, ch_ind = preprocess.get_data("../testData11_14bit_100mV.npy", 10010, 10000)
    
    transformed = np.load("./results/cluster_transformed.npy")[:10000]
    
    plt.scatter(transformed[:, 0], transformed[:, 1], c=test_label, s = 4)
    plt.colorbar()
    plt.show()
    
    selected1 = transformed[:, 1] > 2.0* transformed[:, 0] + 90.0
    selected2 = transformed[:, 1] < 2.0* transformed[:, 0] - 90.0
    selected = np.logical_or(selected1, selected2)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=selected, s = 4)
    plt.colorbar()
    plt.show()
    
    evt_ind_selected = np.array(list(set(evt_ind[:len(test_data)][selected])))
    #evt_ind_selected = np.arange(len(test_data))
    
    
    lifetime("../testData11_14bit_100mV.npy", evt_ind_selected)
    
    
if __name__ == '__main__':
	muon_lifetime()