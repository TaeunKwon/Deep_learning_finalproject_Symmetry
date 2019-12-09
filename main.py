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

tf.enable_eager_execution()

def train(model, train_data):
    
    BSZ = model.batch_size
    
    indices = np.arange(len(train_data))
    indices = tf.random.shuffle(indices)
    shuffled_data = tf.gather(train_data, indices)
    
    curr_loss = 0
    step = 0
    
    #optimizer =  tf.keras.optimizers.Adam(model.learning_rate)
    
    for start, end in zip(range(0, len(shuffled_data) - BSZ, BSZ), range(BSZ, len(shuffled_data), BSZ)):
        with tf.GradientTape() as tape:
            encoded = model.call(shuffled_data[start:end])
            #print("encoding done with ", encoded.get_shape())
            loss = model.loss_function(encoded, shuffled_data[start:end])
        
        curr_loss += loss/BSZ
        step += 1
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))    
        #print(step, loss)
        if step % 10 == 0:
            #print('deconv1d bias:', model.decoder.deconv1_b.numpy())
           # print(gradients)
            print('%dth batches, \tAvg. loss: %.3f' % (step, loss/BSZ))
    return curr_loss/step

def test(model, test_data):
    #encoded = model.call(test_data)
    #loss = model.loss_function(encoded, test_data)
    BSZ = model.batch_size
    curr_loss = 0
    step = 0
    for start, end in zip(range(0, len(test_data) - BSZ, BSZ), range(BSZ, len(test_data), BSZ)):
        encoded = model.call(test_data[start:end])
        loss = model.loss_function(encoded, test_data[start:end])
        
        curr_loss += loss/BSZ
        step += 1  

        if step % 10 == 0:
            print('%dth batches, \tAvg. loss: %.3f' % (step, loss/BSZ))
    
    return curr_loss/step

def train_cluster(model, train_data, train_label):
    
    num_iter = 10#200
    iter_ = 0
    
    for i in range(num_iter):
        print(iter_+1, 'th iteration:')
        with tf.GradientTape() as tape:
            q = model.call(train_data)
            if (iter_ % 40 == 0):
                p = model.target_distribution(q)
        
            loss = model.loss_function(q, p)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))    

        print('loss: %.3f' % (loss))
        iter_ += 1
        
        plt.scatter(q[:, 0], q[:, 1], s = 4, c=train_label)
        plt.show()
        
    return loss


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"autoencoder", "cluster"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [autoencoder/cluster]")
        exit()
    
    pulse_data, label, test_data, test_label, evt_ind, ch_ind = preprocess.get_data("../testData11_14bit_100mV.npy", 5000, 1000)
  
    model = AutoEncoder()
    checkpoint_dir = './checkpoint'
    checkpoint = tf.train.Checkpoint(model = model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    
    if sys.argv[1] == "autoencoder":
        start = time.time()    
        
        num_epochs = 20
        curr_loss = 0
        epoch = 0
        for i in range(num_epochs):
            print(epoch+1, 'th epoch:')
            tot_loss = train(model, pulse_data)
            curr_loss += tot_loss
            epoch += 1
            #print('%dth epochs, \tloss: %.3f' % (epoch, curr_loss / epoch))
            
        print("Test loss:", test(model, test_data))
        print("Process time : {} s".format(int(time.time() - start)))
        print("Saving Checkpoint...")
        manager.save()
        
        #visualization.plot_1evt(pulse_data[733], tf.squeeze(model.call(tf.reshape(pulse_data[733], (1, 1300, 4)))).numpy())
        visualization.plot_1ch(test_data[7], tf.squeeze(model.call(tf.reshape(test_data[7], (1, 1300, 1)))).numpy())
        #visualization.plot_1ch(test_data[33], tf.squeeze(model.call(tf.reshape(test_data[33], (1, 1300, 1)))).numpy())
        #visualization.plot_1ch(test_data[46], tf.squeeze(model.call(tf.reshape(test_data[46], (1, 1300, 1)))).numpy())
        #visualization.plot_1ch(test_data[25], tf.squeeze(model.call(tf.reshape(test_data[25], (1, 1300, 1)))).numpy())
        visualization.feature_v_proj(model, test_data, test_label)
    
    elif sys.argv[1] == "cluster":
        checkpoint.restore(manager.latest_checkpoint)
        visualization.feature_v_proj(model, test_data, test_label)

    model_cluster = clustering(model.encoder)
    kmeans = KMeans(n_clusters = 3, n_init = 20)
    cluster_pred = kmeans.fit_predict(model.encoder(tf.reshape(pulse_data, (-1, 1300,1))))
    model_cluster.cluster.set_weights([kmeans.cluster_centers_])
    
    #num_iter = 50
    #iter_ = 0
    tot_loss = train_cluster(model_cluster, pulse_data, label)
    #for i in range(num_iter):
    #    print(iter_+1, 'th iteration:')
        
    #    iter_ += 1
        
    #    prbs = model_cluster.cluster.call(model_cluster.encoder.call(tf.reshape(pulse_data, (-1, 1300,1))))
    #    plt.scatter(prbs[:, 0], prbs[:, 1], s = 4, c=label)
     #   plt.show()
    
    visualization.feature_v_proj(model_cluster, test_data, test_label)
    
if __name__ == '__main__':
	main()

