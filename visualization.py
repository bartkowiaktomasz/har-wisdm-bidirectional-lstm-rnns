import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw(x,y,z):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlabel('NUMBER OF HIDDEN NEURONS')
    ax.set_ylabel('BATCH SIZE')
    ax.set_zlabel('ACCURACY')
    ax.scatter(x, y, z)

    plt.show()

if __name__ == '__main__':
    dict = np.load('hyperparametersOptimized.npy').item()
    print("Accuracy\tSegment time size\tNo Hidden Neurons\tBatch size")
    for i in range(len(dict['SEGMENT_TIME_SIZE'])):
        print(dict['ACCURACY'][i], "\t", dict['SEGMENT_TIME_SIZE'][i], "\t\t\t\t",\
              dict['N_HIDDEN_NEURONS'][i], "\t\t", dict['BATCH_SIZE'][i])
    a = np.asarray(dict['SEGMENT_TIME_SIZE'])
    b = np.asarray(dict['N_HIDDEN_NEURONS'])
    c = np.asarray(dict['BATCH_SIZE'])
    acc = np.asarray(dict['ACCURACY'])

    draw(b, c, acc)
