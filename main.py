import os
import sys
import numpy as np
import tensorflow as tf
from autoencoder import AutoEncoder
import time
import preprocess
from visualization import plot_1evt

def train(model, train_data):
    
    BSZ = model.batch_size
    
    train_data = np.reshape(train_data, (-1, 2700, 4))
    indices = np.arange(len(train_data))
    indices = tf.random.shuffle(indices)
    shuffled_data = tf.gather(train_data, indices)
    
    curr_loss = 0
    step = 0
    
    optimizer =  tf.keras.optimizers.Adam(model.learning_rate)
    
    for start, end in zip(range(0, len(shuffled_data) - BSZ, BSZ), range(BSZ, len(shuffled_data), BSZ)):
        with tf.GradientTape() as tape:
            encoded = model.call(shuffled_data[start:end])
            #print("encoding done with ", encoded.get_shape())
            loss = model.loss_function(encoded, train_data[start:end])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))    
        
        #print("Trainable parameters:", model.decoder.deconv1_b)
        
        curr_loss += loss
        step += 1

        #print(step, loss)
        #if step % 50 == 0:
        print('%dth batches, \tAvg. loss: %.3f' % (step, curr_loss/step))
    return curr_loss/step


def main():
    pulse_data, label = preprocess.get_data("./testData11_14bit_100mV.npy")
    model = AutoEncoder()
    
    
    start = time.time()    
    
    num_epochs = 1
    curr_loss = 0
    epoch = 0
    
    for i in range(num_epochs):
        tot_loss = train(model, pulse_data)
        curr_loss += tot_loss
        epoch += 1
        print('%dth epochs, \tloss: %.3f' % (epoch, curr_loss / epoch))
        
    print("Final loss:", curr_loss / num_epochs)
    
    print("Process time : {} s".format(int(time.time() - start)))
    
    plot_1evt(pulse_data[733], tf.squeeze(model.call(tf.reshape(pulse_data[733], (1, 2700, 4)))))
    
if __name__ == '__main__':
	main()

