import os
import sys
import numpy as np
import time
import preprocess as preprocess
import lifetime
import matplotlib.pyplot as plt

def muon_lifetime():
    
    pulse_data, label, test_data, test_label, evt_ind, ch_ind = preprocess.get_data("../testData11_14bit_100mV.npy", 5500, 5000)

    transformed = np.load("./results/transformed.npy")
    
    plt.scatter(transformed[:, 0], transformed[:, 1], c=test_label, s = 4)
    plt.colorbar()
    plt.show()
    
    selected = np.logical_and(transformed[:, 1] >35, transformed[:, 1] <85)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=selected, s = 4)
    plt.colorbar()
    plt.show()
    
    evt_ind_selected = np.array(list(set(evt_ind[:len(test_data)][selected])))
    
    lifetime.lifetime("../testData11_14bit_100mV.npy", evt_ind_selected)
    
    
if __name__ == '__main__':
	muon_lifetime()