import os
import matplotlib.pyplot as plt
from model import build_model
from data_loader import load_datasets
from config import *
import tensorflow as tf

def train():
    os.makedirs("saved_model", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ds, val_ds, _ = load_datasets()
    model = build_model()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    )


    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS//2,  # train head for half epochs
        callbacks=[checkpoint, reduce_lr,early_stop]
    )

 
    base_model = model.layers[6] # ImageNet-pretrained CNN حتى نحافظ على اوزان النموذج وندرب بس الراس
    base_model.trainable = True

    for layer in base_model.layers[:80]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, reduce_lr,early_stop]
    )

    acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
    val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']

    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    plt.title("Training Curve")
    plt.savefig(f"{RESULTS_DIR}/training_curve.png")
    plt.close()
