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

LABELS_NAMES = [
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

RANDOM_SEED = 13

# Data preprocessing
TIME_STEP = 100
SEGMENT_TIME_SIZE = 180

# Model
N_CLASSES = 6
N_FEATURES = 3  # x-acceleration, y-acceleration, z-acceleration

# Hyperparameters
N_LSTM_LAYERS = 2
N_EPOCHS = 50
L2_LOSS = 0.0015
LEARNING_RATE = 0.0025
N_HIDDEN_NEURONS = 30
BATCH_SIZE = 64
