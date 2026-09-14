import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import load_datasets
from model import build_model
from config import *


def evaluate():
    _, _, test_ds = load_datasets()

    model = build_model()

    model.load_weights(MODEL_SAVE_PATH)

    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    os.makedirs(RESULTS_DIR, exist_ok=True)


    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()


    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    print(f"Results saved to: {RESULTS_DIR}")