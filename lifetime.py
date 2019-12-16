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
        peakwhere2[i] = max_arg[i] + 220 + 5 if (maxes[i] > 500.) else 0
    
    return peakwhere2
    

def lifetime(delta_t, muon_evt_ind):

    plt.figure(1)
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
