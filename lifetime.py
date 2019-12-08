import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_func(x, a, tau, bkg):
  return a * np.exp(-x/tau) + bkg

def find_peak2(b): # b is (2700, ) array
    numSamples = len(b)
    for i in range(220, numSamples - 5):
        # Scanning a block of 10 samples. If there is a huge jump:
        if max(b[i-5:i+5])-min(b[i-5:i+5]) > 100:
            maxInd = np.argmax(b[i:min(i+50, numSamples)]) +i
            return maxInd
    return 0

def lifetime(filename, muon_evt_ind):
    dt = 0.008 # us
    data = np.load(filename, mmap_mode = 'r')
    muon_data = data[muon_evt_ind][:,0,:]+data[muon_evt_ind][:,1,:]
    # muon_data is shape of (num_muon_data, numSamples)
    num_muon_data = len(muon_data)
    
    peakWhere1 = np.argmax(muon_data[:,190:220], axis = 1) +190
    peakWhere2 = np.zeros((num_muon_data), dtype = np.int16)
    
    for i in range(num_muon_data):
        peakWhere2[i] = find_peak2(muon_data[i])

    delta_t = (peakWhere2 - peakWhere1) * dt

    plt.figure(1)
    hist, bin_edge, patch = plt.hist(delta_t, bins = 20, label = 'data', histtype = 'step')
    
    bin_center = (bin_edge[1:]+bin_edge[:-1])/2
    
    bin_center = bin_center[hist!=0]
    hist = hist[hist!=0]
    print('number of decay events:', sum(hist))
    
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