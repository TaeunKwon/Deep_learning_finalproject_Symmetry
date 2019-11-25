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
    :return: Index of maximum (length 4 array), maximum value (length 4 array)
    '''
    # numChannels = len(a)
    threshold = 100 # mV
    maxInd = np.argmax(a[:,190:210], axis = 1) +190
    # Apply threshold to all channels. If the peak is lower than threshold, index is 0
    maxValue = np.array([a[j, maxInd[j]] for j in range(numChannels)])
    cut_thres = (maxValue > threshold)
    maxInd = maxInd * cut_thres
    return maxInd, maxValue
    
def get_data(filename ='./testData11_14bit_100mV.npy', len_data_to_load = 0, len_test = 0):
    '''
    :param filename: name of file to load
    :return: train data array (size: numEvt x numSam ), train label array 
             (length: numEvt), test data array, test label array
             Label is 1 if ch 0 and 1 have peak.
             Label is 2 if ch 0, 1, and 2 have peak.
             Label is 3 if ch 0, 1, 2, and 3 have peak.
    '''
    data = np.load(filename) # size: numEvt x numCh x numSam 
    numEvents = len(data)
    numChannels = len(data[0])

    peakWhere1 = np.zeros((numEvents,numChannels), dtype = np.int16)
    peakMax1 = np.zeros((numEvents,numChannels), dtype = np.int16)
    label = np.zeros(numEvents, dtype = np.int16)

    for ievt in range(numEvents):
        maxInd, maxValue = find_peak1(data[ievt])
        peakWhere1[ievt,:] = maxInd
        peakMax1[ievt,:] = maxValue
        if maxInd[0] & maxInd[1]:
            if maxInd[2]:
                if maxInd[3]:
                    label[ievt] = 3
                else:
                    label[ievt] = 2
            else:
                label[ievt] = 1
    
    #Flatten           
    if len_data_to_load:
        data = np.reshape(data[:len_data_to_load], (-1,2700))
        peakWhere1 = peakWhere1[:len_data_to_load].flatten()
        peakMax1 = peakMax1[:len_data_to_load].flatten()
        label = np.repeat(label[:len_data_to_load], numChannels)
    else:
        data = np.reshape(np.transpose(data, (0,2,1)), (-1,2700))
        peakWhere1 = peakWhere1.flatten()
        peakMax1 = peakMax1.flatten()
        label = np.repeat(label, numChannels)
    
    #Get rid of zero events
    cut_thres = (peakWhere1 != 0)
    data = data[cut_thres][:,100:1400]
    peakWhere1 = peakWhere1[cut_thres]
    peakMax1 = peakMax1[cut_thres]
    label = label[cut_thres]
    
    data = data / peakMax1[:,None]
                
    if not len_data_to_load:
        len_data_to_load = len(label)
    if len_test:
        num_test = len_test
    else:
        num_test = int(np.ceil(len_data_to_load*0.01))
    
    return data[num_test:len_data_to_load],label[num_test:len_data_to_load],data[:num_test],label[:num_test]