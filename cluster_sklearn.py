from sklearn.manifold import TSNE
import os
import sys
import numpy as np
import tensorflow as tf
from autoencoder import AutoEncoder
import time
import preprocess as preprocess
import visualization
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from lifetime import lifetime

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
            
def test_loss(model, test_data, loss):
    #encoded = model.call(test_data)
    #loss = model.loss_function(encoded, test_data)
    BSZ = model.batch_size
    step = 0
    for start, end in zip(range(0, len(test_data) - BSZ, BSZ), range(BSZ, len(test_data), BSZ)):
        decoded = model.call(test_data[start:end]).numpy()
        loss[start:end] = np.sum((test_data[start:end] - decoded)*(test_data[start:end] - decoded)/tf.square(decoded), axis = 1).reshape((100))
        step+=1
        if step % 10 == 0:
            print('%dth batches done.' % (step))


def cluster_sklearn():    
    pulse_data, label, test_data, test_label, train_evt_ind, test_evt_ind, train_ch_ind, test_ch_ind = preprocess.get_data("../testData11_14bit_100mV.npy",3000,1000)
    #pulse_data, label, test_data, test_label, train_evt_ind, test_evt_ind, train_ch_ind, test_ch_ind = preprocess.get_data("../muon_data_14bit_2.npy", 10010, 10000)
    test_data = pulse_data[100:1100]
    visualization.plot_1evt(test_data[0], test_data[0])
    
    model = AutoEncoder()
    checkpoint_dir = './checkpoint'
    checkpoint = tf.train.Checkpoint(model = model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    
    encoded = np.zeros([len(test_data),10])
    test(model, test_data, encoded)
    #np.save("./results/encoded_50000evts.npy",encoded)
    
    transformed = TSNE(n_components=2).fit_transform(encoded)
    #transformed_nD = TSNE(n_components=4).fit_transform(encoded)
    #transformed_nD = PCA(n_components=5).fit_transform(encoded)
    
    #clustered = DBSCAN(min_samples = 40).fit(transformed)
    #clustered = SpectralClustering(n_clusters = 2,assign_labels="discretize").fit(transformed)
    #clustered = AgglomerativeClustering().fit(transformed_nD)
    clustered = OPTICS(min_samples = 100, min_cluster_size = 100).fit(transformed)
    #clustered = BayesianGaussianMixture(n_components = 3).fit(transformed)
    cluster_label = clustered.labels_
    #cluster_label = clustered.predict(transformed)
 
    #transformed2 = TSNE(n_components=2).fit_transform(transformed)    
    
    plt.figure()
    plt.scatter(transformed[:, 0], transformed[:, 1], c=test_label, s = 4)
    #plt.scatter(transformed[:, 0], transformed[:, 1], s = 4)
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.scatter(transformed[:, 0], transformed[:, 1], c=cluster_label, s = 4)
    #plt.scatter(transformed_nD[:, 0], transformed_nD[:, 1], c=test_label, s = 4)
    plt.colorbar()
    plt.show()
    
    #np.save("./results/transformed_2Vthres_191211.npy",transformed)
    #np.save("./results/cluster_label_OPTICS_2Vthres_191211.npy",cluster_label)
    
def cal_chi2():
    pulse_data, label, test_data, test_label, train_evt_ind, test_evt_ind, train_ch_ind, test_ch_ind = preprocess.get_data("../testData11_14bit_100mV.npy",10010,10000)
    visualization.plot_1ch(test_data[0], test_data[0])
    
    model = AutoEncoder()
    checkpoint_dir = './checkpoint'
    checkpoint = tf.train.Checkpoint(model = model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    
    loss = np.zeros(len(test_data))
    test_loss(model, test_data, loss)
    
    #std = np.reshape(np.std(test_data[:,:100,:], axis = 1),-1)
    chi = np.sqrt(loss)/1300#/std
    plt.figure()
    plt.hist(chi, bins = 20, range = (0,10), histtype = 'step')
    (ymin, ymax) = plt.ylim()
    plt.plot([1.0, 1.0], [ymin, ymax],'--')
    plt.ylim(ymin, ymax)
    plt.xlabel("chi-sqaure")
    plt.ylabel("counts")
    
if __name__ == '__main__':
	cluster_sklearn()
