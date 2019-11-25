import numpy as np
import matplotlib.pyplot as plt

def plot_1evt(en_input, de_output):
    '''
    Plot encoder input and decoder output
    :param en_input: encoder input. 2700 x 4 array
    :param de_output: encoder input. 2700 x 4 array
    '''
    fig = plt.figure(figsize = [8, 10])
    ax = []
    ax.append(fig.add_axes([0.1, 0.7, 0.8, 0.2], xticklabels=[]))
    ax.append(fig.add_axes([0.1, 0.5, 0.8, 0.2], xticklabels=[]))
    ax.append(fig.add_axes([0.1, 0.3, 0.8, 0.2], xticklabels=[]))
    ax.append(fig.add_axes([0.1, 0.1, 0.8, 0.2]))
    
    for i in range(len(en_input[0])):
        ax[i].plot( en_input[:,i],color = 'C0', label = 'ch'+str(i)+' input')
        ax[i].plot(de_output[:,i],'--',color = 'C1', label = 'ch'+str(i)+' output')
        ax[i].set_ylabel('amplitude [mV]')
        ax[i].legend(loc=1)
    
    ax[3].set_xlabel('time [8 ns]')
    