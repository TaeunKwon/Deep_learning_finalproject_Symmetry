import numpy as np
#import matplotlib.pyplot as plt
#from scipy.signal import find_peaks

numEvents = 100000
numChannels = 4
#threshold = -100 #-2000 # mV
dt = 0.008 #us
numSamples = 2700

def find_peak1(a): 
    '''
    Find the index of the first peak.
    The first peak is always around index ~ 200 = pretrigger
    :param a: a is an event. numChannels x numSamples = 4 x 2700 array 
    :return: Index of maximum
    '''
    # numChannels = len(a)
    threshold = 100 # mV
    maxInd = np.argmax(a[:,190:210], axis = 1) +190
    # Apply threshold to all channels. If the peak is lower than threshold, index is 0
    maxInd = maxInd * (np.array([a[j, maxInd[j]] for j in range(numChannels)]) > threshold)
    return maxInd
    
def get_data(filename ='./testData11_14bit_100mV.npy'):
    '''
    :param filename: name of file to load
    :return: data array (size: numEvt x numCh x numSam) and label array 
             (length: numEvt). 
             Label is 1 if ch 0 and 1 have peak.
             Label is 2 if ch 0, 1, and 2 have peak.
             Label is 3 if ch 0, 1, 2, and 3 have peak.
    '''
    data = np.load(filename, mmap_mode = 'r')
    
    numEvents = len(data)
    numChannels = len(data[0])
    peakWhere1 = np.zeros((numEvents,numChannels), dtype = np.int16)
    label = np.zeros(numEvents, dtype = np.int16)
#    peakHeight1 = np.zeros((numEvents,numChannels), dtype = np.int16)

    for ievt in range(numEvents):
        maxInd = find_peak1(data[ievt])
        peakWhere1[ievt,:] = maxInd
        if maxInd[0] & maxInd[1]:
            if maxInd[2]:
                if maxInd[3]:
                    label[ievt] = 3
                else:
                    label[ievt] = 2
            else:
                label[ievt] = 1
    data = np.transpose(np.float32(data), axes = (0,2,1))
    return data, label
    
    
if __name__ == '__main__':
    reduce_data()