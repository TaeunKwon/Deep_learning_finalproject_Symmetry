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

#tf.enable_eager_execution()

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
    
    if num_iter == 0:
        shuffled_data = train_data
        shuffled_ch_ind = ch_ind
        shuffled_delta_t = delta_t
    else:
        indices = np.arange(len(train_data))
        indices = tf.random.shuffle(indices)
        origin_indices = tf.argsort(indices)
        shuffled_data = tf.gather(train_data, indices)
        p = tf.gather(p, indices)
        shuffled_ch_ind = tf.gather(ch_ind, indices)
        shuffled_delta_t = tf.gather(delta_t, indices)
    
    curr_loss = 0
    step = 0      
    for start, end in zip(range(0, len(shuffled_data) - BSZ, BSZ), range(BSZ, len(shuffled_data), BSZ)):
        with tf.GradientTape(persistent = True) as tape:
            q = model.call(shuffled_data[start:end])
            
            if (num_iter % 30) == 0:
                if start == 0 :
                    p = model.target_distribution(q)
                else:
                    p = tf.concat([p, model.target_distribution(q)], axis = 0)
            loss_cluster = model.loss_function(q, p[start:end],shuffled_delta_t[start:end], shuffled_ch_ind[start:end])

            if (num_iter % 5) == 0:
                encoded = model_auto.call(shuffled_data[start:end])
                loss_auto = model_auto.loss_function(encoded, shuffled_data[start:end])
            
        gradients_cluster = tape.gradient(loss_cluster, model.trainable_variables)        
        model.optimizer.apply_gradients(zip(gradients_cluster, model.trainable_variables))  
        
        step += 1 
        
        if (num_iter % 5) == 0:
            gradients_auto = tape.gradient(loss_auto, model_auto.trainable_variables) 
            model_auto.optimizer2.apply_gradients(zip(gradients_auto, model_auto.trainable_variables)) 
            if step % 2 == 0:
                print('%dth batches, \tloss_auto: %.3f' % (step, loss_auto/BSZ))
        curr_loss += loss_cluster
        
        if step % 2 == 0:
            print('%dth batches, \tloss_cluster: %.7f' % (step, loss_cluster/BSZ))
            
    with tf.GradientTape(persistent = True) as tape:
        q = model.call(shuffled_data[end:])
            
        if  (num_iter % 30) == 0:
            p = tf.concat([p, model.target_distribution(q)], axis = 0)
        loss_cluster = model.loss_function(q, p[end:], shuffled_delta_t[end:], shuffled_ch_ind[end:])
            
        if (num_iter % 5) == 0:
            encoded = model_auto.call(shuffled_data[end:])  
            loss_auto = model_auto.loss_function(encoded, shuffled_data[end:])
            
    gradients_cluster = tape.gradient(loss_cluster, model.trainable_variables)        
    model.optimizer.apply_gradients(zip(gradients_cluster, model.trainable_variables))  
    if (num_iter % 5) == 0:
        gradients_auto = tape.gradient(loss_auto, model_auto.trainable_variables) 
        model_auto.optimizer2.apply_gradients(zip(gradients_auto, model_auto.trainable_variables)) 
        
        curr_loss += loss_cluster
        step += 1 
        
    if num_iter != 0:
        p = tf.gather(p, origin_indices)
    return curr_loss/step, p

def lifetime_calc(model, encoder, train_data, delta_t, evt_ind, ch_ind):
    BSZ = model.batch_size
    
    for start, end in zip(range(0, len(train_data) - BSZ, BSZ), range(BSZ, len(train_data), BSZ)):
        if start == 0:
            q = model.call(train_data[start:end])
        else:
            q = tf.concat([q, model.call(train_data[start:end])], axis = 0)
    q = tf.concat([q, model.call(train_data[end:])], axis = 0)
    
    ind = tf.argmax(q, axis = 1)
    
    print(tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(ch_ind, 0), tf.equal(ind, 1)), dtype = tf.int32)))
    print(tf.reduce_sum(tf.cast(tf.not_equal(ch_ind, 0), dtype = tf.int32)))
    
    print(tf.reduce_sum(tf.cast(tf.equal(ch_ind, 0), dtype = tf.int32)))
    print(tf.reduce_sum(tf.cast(tf.equal(ch_ind, 1), dtype = tf.int32)))
    print(tf.reduce_sum(tf.cast(tf.equal(ch_ind, 2), dtype = tf.int32)))
    print(tf.reduce_sum(tf.cast(tf.equal(ch_ind, 3), dtype = tf.int32)))
    
    num_bkgcluster = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(ch_ind, 0), tf.not_equal(ind, 2)), dtype = tf.float32))
    num_bkg = tf.reduce_sum(tf.cast(tf.not_equal(ch_ind, 0), dtype = tf.float32))
            
    print('Accuracy: %f' % (tf.cast(num_bkgcluster/num_bkg, dtype = tf.float32)))
    
    num_bkgcluster2 = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(ch_ind, 0), tf.equal(ind, 2)), dtype = tf.float32))
    num_bkg2 = tf.reduce_sum(tf.cast(tf.equal(ch_ind, 0), dtype = tf.float32))
            
    print('Accuracy: %f' % (tf.cast(num_bkgcluster2/num_bkg2, dtype = tf.float32)))
    
    visualization.feature_v_proj(encoder, train_data, ch_ind)
    visualization.feature_v_proj(encoder, train_data, ind)
    
    cluster1 = np.array(list(set(evt_ind[-len(ind):][ind == 2])))
    cluster2 = np.array(list(set(evt_ind[-len(ind):][ind == 1])))
    cluster3 = np.array(list(set(evt_ind[-len(ind):][ind == 0])))
    
    #lifetime("../testData11_14bit_100mV.npy", np.concatenate((cluster1, cluster3), axis = 0))
    lifetime(delta_t[cluster1], cluster1)
    lifetime(delta_t[cluster2], cluster2)
    lifetime(delta_t[cluster3], cluster3)
    
    cluster = np.concatenate([cluster2,cluster3])
    
    lifetime(delta_t[cluster], cluster)
    

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"autoencoder", "cluster", "lifetime"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [autoencoder/cluster/lifetime]")
        return
    
    #pulse_data, label, test_data, test_label, train_evt_ind, test_evt_ind, train_ch_ind, test_ch_ind = preprocess.get_data("../testData11_14bit_100mV.npy")
    pulse_data1, label1, test_data1, test_label1, train_evt_ind1, test_evt_ind1, train_ch_ind1, test_ch_ind1 = preprocess.get_data("../DL_additional_data/muon_data_deep_learning_0_1.npy")
    print('first data loaded')
    pulse_data2, label2, test_data2, test_label2, train_evt_ind2, test_evt_ind2, train_ch_ind2, test_ch_ind2 = preprocess.get_data("../DL_additional_data/muon_data_deep_learning_0_2.npy")
    print('second data loaded')
    pulse_data3, label3, test_data3, test_label3, train_evt_ind3, test_evt_ind3, train_ch_ind3, test_ch_ind3 = preprocess.get_data("../DL_additional_data/muon_data_deep_learning_1_1.npy")
    print('third data loaded')
    pulse_data4, label4, test_data4, test_label4, train_evt_ind4, test_evt_ind4, train_ch_ind4, test_ch_ind4 = preprocess.get_data("../DL_additional_data/muon_data_deep_learning_1_2.npy")
    print('fourth data loaded')
    pulse_data5, label5, test_data5, test_label5, train_evt_ind5, test_evt_ind5, train_ch_ind5, test_ch_ind5 = preprocess.get_data("../DL_additional_data/muon_data_deep_learning_2_1.npy")
    print('data loading finished')
    
    pulse_data = np.concatenate([pulse_data1, pulse_data2, pulse_data3, pulse_data4, pulse_data5])
    del pulse_data1, pulse_data2, pulse_data3, pulse_data4, pulse_data5
    
    label =  np.concatenate([label1, label2, label3, label4, label5])
    del label1, label2, label3, label4, label5
    
    test_data =  np.concatenate([test_data1, test_data2, test_data3, test_data4, test_data5])
    del test_data1, test_data2, test_data3, test_data4, test_data5
    
    test_label =  np.concatenate([test_label1, test_label2, test_label3, test_label4, test_label5])
    del test_label1, test_label2, test_label3, test_label4, test_label5
    
    train_evt_ind =  np.concatenate([train_evt_ind1, train_evt_ind2, train_evt_ind3, train_evt_ind4, train_evt_ind5])
    del train_evt_ind1, train_evt_ind2, train_evt_ind3, train_evt_ind4, train_evt_ind5
    
    #test_evt_ind = np.concat([test_evt_ind1, test_evt_ind2, test_evt_ind3, test_evt_ind4, test_evt_ind5])
    del test_evt_ind1, test_evt_ind2, test_evt_ind3, test_evt_ind4, test_evt_ind5
    
    train_ch_ind =  np.concatenate([train_ch_ind1, train_ch_ind2, train_ch_ind3, train_ch_ind4, train_ch_ind5])
    del train_ch_ind1, train_ch_ind2, train_ch_ind3, train_ch_ind4, train_ch_ind5
    
    #test_ch_ind = np.concat([test_ch_ind1, test_ch_ind2, test_ch_ind3, test_ch_ind4, test_ch_ind5])
    del test_ch_ind1, test_ch_ind2, test_ch_ind3, test_ch_ind4, test_ch_ind5
    
    #delta_t_origin = preprocess.get_delta_t("../testData11_14bit_100mV.npz", train_evt_ind)
    train_delta_t1, test_delta_t1 = preprocess.get_delta_t2("../DL_additional_data/muon_data_deep_learning_0_1.npz")#delta_t_origin[train_evt_ind, train_ch_ind]
    train_delta_t2, test_delta_t2 = preprocess.get_delta_t2("../DL_additional_data/muon_data_deep_learning_0_2.npz")
    train_delta_t3, test_delta_t3 = preprocess.get_delta_t2("../DL_additional_data/muon_data_deep_learning_1_1.npz")
    train_delta_t4 , test_delta_t4 = preprocess.get_delta_t2("../DL_additional_data/muon_data_deep_learning_1_2.npz")
    train_delta_t5, test_delta_t5 = preprocess.get_delta_t2("../DL_additional_data/muon_data_deep_learning_2_1.npz")
    
    train_delta_t =  np.concatenate([train_delta_t1, train_delta_t2, train_delta_t3, train_delta_t4, train_delta_t5])
    del train_delta_t1, train_delta_t2, train_delta_t3, train_delta_t4, train_delta_t5
    
    #test_delta_t = np.concat([test_delta_t1, test_delta_t2, test_delta_t3, test_delta_t4, test_delta_t5])
    del test_delta_t1, test_delta_t2, test_delta_t3, test_delta_t4, test_delta_t5
    
    print(np.sum(train_delta_t))
    #print(np.sum(test_delta_t))
    
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
        #checkpoint_cluster.restore(manager_cluster.latest_checkpoint)
        
        kmeans = KMeans(n_clusters = 3, init = 'k-means++', n_init = 20, max_iter = 400)
        cluster_pred = kmeans.fit_predict(model.encoder(tf.reshape(pulse_data[:min(len(pulse_data), 10000)], (-1, 1300,1))))
        model_cluster.cluster.set_weights([kmeans.cluster_centers_])
        #print(model_cluster.cluster.clusters)
        #print(kmeans.cluster_centers_)
        #print(kmeans.inertia_)
        
        num_iter = 30
        cnt_iter = 0
        
        p = None
        
        for i in range(num_iter):
            print(cnt_iter+1, 'th iteration:')
            tot_loss, p = train_cluster(model_cluster, model, pulse_data, cnt_iter, p, train_ch_ind, train_delta_t)
            cnt_iter += 1
            prbs = model_cluster.call(tf.cast(tf.reshape(pulse_data[:10000], (-1, 1300,1)), dtype = tf.float32))
            ind = tf.argmax(prbs, axis = 1)
            visualization.feature_v_proj(model.encoder, pulse_data[:10000], ind)
            
            num_bkgcluster = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(train_ch_ind[:10000], 0), tf.not_equal(ind, 0)), dtype = tf.float32))
            num_bkg = tf.reduce_sum(tf.cast(tf.not_equal(train_ch_ind[:10000], 0), dtype = tf.float32))
            
            print('%dth epochs, \tAccuracy: %f' % (cnt_iter+1, tf.cast(num_bkgcluster/num_bkg, dtype = tf.float32)))
            if cnt_iter % 10 == 0 and cnt_iter != 0 :
                print("Saving Checkpoint...")
                manager_cluster.save()
            
            
            
        
        visualization.feature_v_proj(model.encoder, test_data, test_label)
        manager.save()
    
    elif sys.argv[1] == "lifetime":
        checkpoint.restore(manager.latest_checkpoint)
        checkpoint_cluster.restore(manager_cluster.latest_checkpoint)
        lifetime_calc(model_cluster, model.encoder, pulse_data, train_delta_t, train_evt_ind, train_ch_ind)
        
    
if __name__ == '__main__':
	main()

