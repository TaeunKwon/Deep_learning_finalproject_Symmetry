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

def train_cluster(model, model_auto, train_data, num_iter, p, ch_ind, delta_t):
    
    BSZ = model.batch_size
    num_batch = int(len(train_data)/BSZ)
    
    indices = np.arange(len(train_data))
    indices = tf.random.shuffle(indices)
    train_data = tf.reshape(train_data, (-1, 1300,1))
    shuffled_data = tf.gather(train_data, indices)
    shuffled_ch_ind = tf.gather(ch_ind, indices)
    shuffled_delta_t = tf.gather(delta_t, indices)
    
    curr_loss = 0
    step = 0      
    for start, end in zip(range(0, len(shuffled_data) - BSZ, BSZ), range(BSZ, len(shuffled_data), BSZ)):
        with tf.GradientTape(persistent = True) as tape:
            q = model.call(shuffled_data[start:end])
            encoded = model_auto.call(shuffled_data[start:end])
            #if (num_iter % 3 == 0):
            if start == 0:
                p = model.target_distribution(q)
            else:
                p = tf.concat([p, model.target_distribution(q)], axis = 0)
                    
            loss_cluster = model.loss_function(q, p[start:end],shuffled_delta_t[start:end], shuffled_ch_ind[start:end], alpha = 1.5)
            loss_auto = model_auto.loss_function(encoded, shuffled_data[start:end])
        
        gradients_cluster = tape.gradient(loss_cluster, model.trainable_variables)        
        gradients_auto = tape.gradient(loss_auto, model_auto.trainable_variables) 
        model.optimizer.apply_gradients(zip(gradients_cluster, model.trainable_variables))  
        model_auto.optimizer.apply_gradients(zip(gradients_auto, model_auto.trainable_variables)) 
        
        curr_loss += loss_cluster
        step += 1 
        
        if step % 10 == 0:
            print('%dth batches, \tloss_cluster: %.7f, \tloss_auto: %.3f' % (step, loss_cluster, loss_auto))

    return curr_loss/step, p

def lifetime_calc(model, encoder, train_data, evt_ind):
    BSZ = model.batch_size
    
    for start, end in zip(range(0, len(train_data) - BSZ, BSZ), range(BSZ, len(train_data), BSZ)):
        if start == 0:
            q = model.call(encoder(train_data[start:end]))
        else:
            q = tf.concat([q, model.call(encoder(train_data[start:end]))], axis = 0)
    q = tf.concat([q, model.call(encoder(train_data[end:]))], axis = 0)
    
    ind = tf.argmax(q, axis = 1)
    
    visualization.feature_v_proj(encoder, train_data, ind)
    
    cluster1 = np.array(list(set(evt_ind[ind == 1])))
    cluster2 = np.array(list(set(evt_ind[ind == 0])))
    
    lifetime("../testData11_14bit_100mV.npy", cluster1)
    lifetime("../testData11_14bit_100mV.npy", cluster2)
    

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"autoencoder", "cluster", "lifetime"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [autoencoder/cluster]")
        return
    
    pulse_data, label, test_data, test_label, train_evt_ind, test_evt_ind, train_ch_ind, test_ch_ind = preprocess.get_data("../testData11_14bit_100mV.npy", 21000, 1000)
    delta_t_origin = preprocess.get_delta_t("../testData11_14bit_100mV_reduced.npz")
    train_delta_t = delta_t_origin[train_evt_ind, train_ch_ind]
    test_delta_t = delta_t_origin[test_evt_ind, test_ch_ind]
  
    model = AutoEncoder()
    checkpoint_dir = './checkpoint'
    checkpoint = tf.train.Checkpoint(model = model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    
    
    if sys.argv[1] == "autoencoder":
        start = time.time()    
        
        num_epochs = 1
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
        
        visualization.plot_1ch(test_data[7], tf.squeeze(model.call(tf.reshape(test_data[7], (1, 1300, 1)))).numpy())
        visualization.plot_1ch(test_data[33], tf.squeeze(model.call(tf.reshape(test_data[33], (1, 1300, 1)))).numpy())
        visualization.plot_1ch(test_data[46], tf.squeeze(model.call(tf.reshape(test_data[46], (1, 1300, 1)))).numpy())
        visualization.plot_1ch(test_data[25], tf.squeeze(model.call(tf.reshape(test_data[25], (1, 1300, 1)))).numpy())
        visualization.feature_v_proj(model.encoder, test_data, test_label)
    
    elif sys.argv[1] == "cluster":
        checkpoint.restore(manager.latest_checkpoint)
        visualization.feature_v_proj(model.encoder, test_data, test_label)

    model_cluster = clustering(model.encoder)
    checkpoint_dir_cluster = './checkpoint_cluster'
    checkpoint_cluster = tf.train.Checkpoint(model = model_cluster)
    manager_cluster = tf.train.CheckpointManager(checkpoint_cluster, checkpoint_dir_cluster, max_to_keep=3)

    if sys.argv[1] == "cluster":   
        kmeans = KMeans(n_clusters = 2, n_init = 20)
        cluster_pred = kmeans.fit_predict(model.encoder(tf.reshape(pulse_data[:10000], (-1, 1300,1))))
        model_cluster.cluster.set_weights([kmeans.cluster_centers_])
        
        num_iter = 10#20
        cnt_iter = 0
        
        p = None
        
        for i in range(num_iter):
            print(cnt_iter+1, 'th iteration:')
            tot_loss, p = train_cluster(model_cluster, model, pulse_data, cnt_iter, p, train_ch_ind, train_delta_t)
            cnt_iter += 1
            prbs = model_cluster.call(tf.cast(tf.reshape(pulse_data[:10000], (-1, 1300,1)), dtype = tf.float32))
            ind = tf.argmax(prbs, axis = 1)
            visualization.feature_v_proj(model.encoder, pulse_data[:10000], ind)
        
        
        
        visualization.feature_v_proj(model.encoder, test_data, test_label)
        print("Saving Checkpoint...")
        manager_cluster.save()
        manager.save()
    
    elif sys.argv[1] == "lifetime":
        checkpoint.restore(manager.latest_checkpoint)
        checkpoint_cluster.restore(manager_cluster.latest_checkpoint)
        lifetime_calc(model_cluster, model.encoder, pulse_data, train_evt_ind)
    
if __name__ == '__main__':
	main()

