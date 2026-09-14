import os
import tensorflow as tf
from tensorflow.keras import layers, models


SEED = 42
tf.random.set_seed(SEED)


DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_SAVE_PATH = "saved_model/best_model.keras"
RESULTS_DIR = "results"


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CHANNELS = 3  # convert grayscale → RGB
NUM_CLASSES = 5


EPOCHS = 25
LEARNING_RATE = 1e-4


CLASS_NAMES = ['Cavity', 'Fillings', 'Impacted Tooth', 'Implant', 'Normal']


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05),
])
