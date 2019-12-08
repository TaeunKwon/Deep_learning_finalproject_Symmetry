import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    
def plot_1ch(en_input, de_output):
    '''
    Plot encoder input and decoder output
    :param en_input: encoder input. 2700 array
    :param de_output: encoder input. 2700 array
    '''
    plt.figure(figsize = [8, 3])
    plt.plot( en_input,color = 'C0', label = 'input')
    plt.plot(de_output,'--',color = 'C1', label = 'output')
    plt.ylabel('amplitude [mV]')
    plt.legend(loc=1)
    
    plt.xlabel('time [8 ns]')
    
    plt.show()
    
def feature_v_proj(model, data, label):
    NUM_SAMPLES = min(1000, len(label))

    t_data = data[:NUM_SAMPLES]
    t_labels = label[:NUM_SAMPLES]
    z = model.encoder.call(np.reshape(t_data, (-1,1300,1)))
    
    tsne = TSNE(n_components=2)
    transformed = tsne.fit_transform(z)
    colors = t_labels
    plt.scatter(transformed[:, 0], transformed[:, 1], c=colors, s = 4)
    plt.colorbar()
    plt.show()
        