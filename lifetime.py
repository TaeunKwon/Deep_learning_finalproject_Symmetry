import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tensorflow as tf

def exp_func(x, a, tau, bkg):
  return a * np.exp(-x/tau) + bkg

def find_peak2(b, peakwhere2): # b is (2700, ) array
    b = tf.cast(tf.reshape(b, (-1, 2700,1,1)), dtype = tf.float32)

    maxpooled = tf.nn.max_pool(b[:, 220:, :], 10, 1, padding = 'SAME')
    minpooled = -tf.nn.max_pool(-b[:, 220:, :], 10, 1, padding = 'SAME')
    
    diff = tf.reshape(maxpooled - minpooled, [-1, 2480])
    maxes = tf.reduce_max(diff, axis = 1)
    max_arg = tf.argmax(diff, axis = 1)
    
    
    for i in range(len(max_arg)):
        peakwhere2[i] = max_arg[i] + 220 if (maxes[i] > 500.) else 0
    
    return peakwhere2
    
    #for i in range(220, numSamples - 5):
    #    # Scanning a block of 10 samples. If there is a huge jump:
    #    if max(b[i-5:i+5])-min(b[i-5:i+5]) > 100:
    #        maxInd = np.argmax(b[i:min(i+50, numSamples)]) +i
    #        return maxInd
    #return 0


def lifetime(filename, muon_evt_ind):
    dt = 0.008 # us
    data = np.load(filename, mmap_mode = 'r')
    muon_data_0 = data[muon_evt_ind][:,0,:]#+data[muon_evt_ind][:,1,:]
    muon_data_1 = data[muon_evt_ind][:,1,:]
    muon_data_2 = data[muon_evt_ind][:,2,:]
    muon_data_3 = data[muon_evt_ind][:,3,:]
    # muon_data is shape of (num_muon_data, numSamples)

    peakWhere1_0 = np.argmax(muon_data_0[:,190:220], axis = 1) +190
    peakWhere1_1 = np.argmax(muon_data_1[:,190:220], axis = 1) +190
    peakWhere1_2 = np.argmax(muon_data_2[:,190:220], axis = 1) +190
    peakWhere1_3 = np.argmax(muon_data_3[:,190:220], axis = 1) +190
    
    peakWhere2_0 = np.zeros((len(muon_data_0)), dtype = np.int16)
    peakWhere2_1 = np.zeros((len(muon_data_1)), dtype = np.int16)
    peakWhere2_2 = np.zeros((len(muon_data_2)), dtype = np.int16)
    peakWhere2_3 = np.zeros((len(muon_data_3)), dtype = np.int16)
    
    peakWhere2_0 = find_peak2(muon_data_0, peakWhere2_0)
    peakWhere2_1 = find_peak2(muon_data_1, peakWhere2_1)
    peakWhere2_2 = find_peak2(muon_data_2, peakWhere2_2)
    peakWhere2_3 = find_peak2(muon_data_3, peakWhere2_3)

    peakWhere1 = np.zeros((len(muon_data_0)), dtype = np.int16)
    peakWhere2 = np.zeros((len(muon_data_0)), dtype = np.int16)

    for i in range(len(peakWhere2_0)):
        if peakWhere2_0[i] != 0 :
            peakWhere2[i] = peakWhere2_0[i]
            peakWhere1[i] = peakWhere1_0[i]
        elif peakWhere2_1[i] != 0 :
            peakWhere2[i] = peakWhere2_1[i]
            peakWhere1[i] = peakWhere1_1[i]
        elif peakWhere2_2[i] != 0 :
            peakWhere2[i] = peakWhere2_2[i]
            peakWhere1[i] = peakWhere1_2[i]
        elif peakWhere2_3[i] != 0 :
            peakWhere2[i] = peakWhere2_3[i]
            peakWhere1[i] = peakWhere1_3[i]

    delta_t = (peakWhere2 - peakWhere1) * dt

    plt.figure()
    hist, bin_edge, patch = plt.hist(delta_t, bins = 20, range = (0.01, 20.), label = 'data', histtype = 'step')
    print(hist, bin_edge)
    
    bin_center = (bin_edge[1:]+bin_edge[:-1])/2
    
    bin_center = bin_center[hist!=0]
    hist = hist[hist!=0]
    print('number of decay events:', sum(hist), " over ", len(muon_evt_ind))
    
    x = np.linspace(bin_edge[0], bin_edge[-1], 50)
    
    p0 = np.array([hist[0], 2.2, 10])
    popt, pcov = curve_fit(exp_func, bin_center, hist, p0 = p0, sigma = np.sqrt(hist))
    fit_result = exp_func(x, *popt)
    print(popt, np.sqrt(np.diag(pcov)))
    
    print('lifetime =', round(popt[1], 3), '+-', round(np.sqrt(pcov[1][1]), 3), 'us')
    
    
    plt.plot(x, fit_result, '--', label = 'fit result', color = 'C1')
    # points with error bar
    yerr = np.sqrt(hist)
    plt.errorbar(bin_center, hist, yerr=yerr, fmt=',', color = 'black')
    plt.text(0.77, 0.78, 'Livetime 22 hr', transform=plt.gca().transAxes)
    
    plt.xlabel('time [us]')
    plt.ylabel('counts')
    plt.yscale('log')
    plt.legend()
    
    plt.show()
