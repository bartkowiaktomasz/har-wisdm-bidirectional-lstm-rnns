import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.python.tools import inspect_checkpoint as chkp

# Local libraries
from preprocessing import get_convoluted_data
from HAR_Recognition import createBidirLSTM

##################################################
### GLOBAL VARIABLES
##################################################
COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

LABELS = [
    'Downstairs',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Walking'
]

DATA_PATH = 'data/WISDM_ar_v1.1_raw.txt'
MODEL_META_PATH = 'model/classificator.ckpt.meta'
MODEL_CHECKPOINT_PATH = 'model/'

# Model
N_CLASSES = 6
N_FEATURES = 3  # x-acceleration, y-acceleration, z-acceleration

N_HIDDEN_NEURONS = 30

RANDOM_SEED = 13
SEGMENT_TIME_SIZE = 180
TIME_STEP = 100

##################################################
### FUNCTIONS
##################################################

def evaluate(model_meta_path, model_checkpoint_path, X_test, y_test):

    # Display all variables
    # chkp.print_tensors_in_checkpoint_file("model/classificator.ckpt", tensor_name='', all_tensors=False, all_tensor_names=True)

    # Load the parameters
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(model_checkpoint_path))

        # Get the graph saved
        graph = tf.get_default_graph()

        X = graph.get_tensor_by_name("X:0")
        y = graph.get_tensor_by_name("y:0")

        y_pred_softmax = graph.get_tensor_by_name('y_pred_softmax:0')
        correct_pred = tf.equal(tf.argmax(y_pred_softmax, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

        acc = sess.run(accuracy, feed_dict={X: X_test, y: y_test})

        return acc


##################################################
### MAIN
##################################################
if __name__ == '__main__':

    data_convoluted, labels = get_convoluted_data(DATA_PATH, COLUMN_NAMES, SEGMENT_TIME_SIZE, TIME_STEP)

    # SPLIT INTO TRAINING AND TEST SETS
    _, X_test, _, y_test = train_test_split(data_convoluted, labels, test_size=0.3, random_state=RANDOM_SEED)

    accuracy = evaluate(MODEL_META_PATH, MODEL_CHECKPOINT_PATH, X_test, y_test)
    print("Final accuracy: ", accuracy)
