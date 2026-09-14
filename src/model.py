import tensorflow as tf
from tensorflow.keras import layers, models
from config import *
from data_loader import *

def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),  # 3 channels after grayscale->RGB
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # initially freeze

    inputs = layers.Input(shape=(224, 224, 1))
    x = layers.CenterCrop(200, 200)(inputs)
    x = layers.Resizing(224, 224)(x)

    # Preprocessing: normalize and convert to RGB
    x = layers.Lambda(normalize_xray)(x)
    x = layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(x)

    # Data augmentation
    x = data_augmentation(x)

    # Base model
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu6")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
