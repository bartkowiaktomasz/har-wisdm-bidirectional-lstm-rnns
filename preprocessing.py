import numpy as np
import pandas as pd

from scipy import stats

# Returns a tuple consisting of a convoluted data and labels
def get_convoluted_data(data, segment_time_size, time_step):

    data_convoluted = []
    labels = []

    # Slide a "segment_time_size" wide window with a step size of "time_step"
    # print("segment_time_size, N_HIDDEN_NEURONS, BATCH_SIZE: ", segment_time_size, N_HIDDEN_NEURONS, BATCH_SIZE)
    for i in range(0, len(data) - segment_time_size, time_step):
        x = data['x-axis'].values[i: i + segment_time_size]
        y = data['y-axis'].values[i: i + segment_time_size]
        z = data['z-axis'].values[i: i + segment_time_size]
        data_convoluted.append([x, y, z])

        # Label for a data window is the label that appears most commonly
        label = stats.mode(data['activity'][i: i + segment_time_size])[0][0]
        labels.append(label)

    # Convert to numpy
    data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

    # One-hot encoding
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)

    return data_convoluted, labels
