import os
import sys
import numpy as np
import tensorflow as tf
from autoencoder_1dtest import AutoEncoder
import time
import preprocess_1dtest as preprocess
import visualization

tf.enable_eager_execution()

def train(model, train_data):
    
    BSZ = model.batch_size
    
    indices = np.arange(len(train_data))
    indices = tf.random.shuffle(indices)
    train_data = tf.reshape(train_data, (-1, 1300,1))
    shuffled_data = tf.gather(train_data, indices)
    
    curr_loss = 0
    step = 0
    
    #optimizer =  tf.keras.optimizers.Adam(model.learning_rate)
    
    for start, end in zip(range(0, len(shuffled_data) - BSZ, BSZ), range(BSZ, len(shuffled_data), BSZ)):
        with tf.GradientTape() as tape:
            encoded = model.call(shuffled_data[start:end])
            #print("encoding done with ", encoded.get_shape())
            loss = model.loss_function(encoded, train_data[start:end])
        
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
    test_data = tf.reshape(test_data, (-1, 1300,1))

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


def main():
    pulse_data, label, test_data, test_label = preprocess.get_data("../testData11_14bit_100mV.npy", 5000, 1000)
    model = AutoEncoder()
    
    
    start = time.time()    
    
    num_epochs = 10
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
    
    #visualization.plot_1evt(pulse_data[733], tf.squeeze(model.call(tf.reshape(pulse_data[733], (1, 1300, 4)))).numpy())
    visualization.plot_1ch(test_data[7], tf.squeeze(model.call(tf.reshape(test_data[7], (1, 1300, 1)))).numpy())
    visualization.plot_1ch(test_data[33], tf.squeeze(model.call(tf.reshape(test_data[33], (1, 1300, 1)))).numpy())
    visualization.plot_1ch(test_data[46], tf.squeeze(model.call(tf.reshape(test_data[46], (1, 1300, 1)))).numpy())
    visualization.plot_1ch(test_data[25], tf.squeeze(model.call(tf.reshape(test_data[25], (1, 1300, 1)))).numpy())

    visualization.feature_v_proj(model, test_data, test_label)

if __name__ == '__main__':
	main()

