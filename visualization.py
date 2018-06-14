import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw(x,y,z):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlabel('SEGMENT_TIME_SIZE')
    ax.set_ylabel('N_HIDDEN_NEURONS')
    ax.set_zlabel('ACCURACY')
    ax.scatter(x, y, z)

    plt.show()

if __name__ == '__main__':
    dict = np.load('hyperparametersOptimized.npy').item()

    a = np.asarray(dict['SEGMENT_TIME_SIZE'])
    b = np.asarray(dict['N_HIDDEN_NEURONS'])
    c = np.asarray(dict['BATCH_SIZE'])
    out = np.asarray(dict['ACCURACY'])

    draw(x, y, z)
