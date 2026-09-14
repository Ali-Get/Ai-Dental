import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.saving import register_keras_serializable
from config import *

@register_keras_serializable()
def normalize_xray(x):
    x = tf.cast(x, tf.float32)
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    return (x - min_val) / (max_val - min_val + 1e-6)

def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        seed=SEED
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        seed=SEED
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        shuffle=False
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    return train_ds, val_ds, test_ds
