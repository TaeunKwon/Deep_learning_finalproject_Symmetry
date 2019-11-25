import os
import sys
import numpy as np
import tensorflow as tf
from autoencoder import AutoEncoder
import time
import preprocess


def train(model, train_data):
    
    BSZ = model.batch_size
    
    indices = np.arange(len(train_data))
    indices = tf.random.shuffle(indices)
    shuffled_data = tf.gather(train_data, indices)
    
    curr_loss = 0
    step = 0
    
    for start, end in zip(range(0, len(shuffled_data) - BSZ, BSZ), range(BSZ, len(shuffled_data), BSZ)):
        with tf.GradientTape() as tape:
            encoded = model.call(shuffled_data[start:end])
            print("encoding done with ", encoded.get_shape())
            loss = model.loss_function(encoded, train_data[start:end])
        
        curr_loss += loss
        step += 1
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))    
        
        #if step % 100 == 0:
        print('%dth batches, \tAvg. loss: %.3f' % (step, tf.exp(curr_loss/step)))
    return curr_loss/step


def main():
    pulse_data, label = preprocess.get_data("./testData11_14bit_100mV.npy")
    model = AutoEncoder()
    
    
    start = time.time()    
    
    num_epochs = 10
    curr_loss = 0
    epoch = 0
    for i in range(num_epochs):
        tot_loss = train(model, pulse_data)
        curr_loss += tot_loss
        epoch += 1
        print('%dth epochs, \tAccuracy: %.3f' % (epoch, curr_loss / epoch))
        
    print("Test Accuracy:", curr_loss / num_epochs)
    
    print("Process time : {} s".format(int(time.time() - start)))
    
if __name__ == '__main__':
	main()

