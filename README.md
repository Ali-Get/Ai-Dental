# AI Dental X-ray Classification

A deep learning system for classifying dental X-ray images into five categories using TensorFlow and MobileNetV2.

## Overview

This project implements a deep learning pipeline for the classification of dental X-ray images.

The model classifies dental X-ray images into the following five categories:

- Cavity
- Fillings
- Impacted Tooth
- Implant
- Normal

The system uses a grayscale X-ray input pipeline and converts the images to RGB internally to use the ImageNet-pretrained MobileNetV2 architecture.

## Model Architecture

The project uses:

- MobileNetV2 pretrained on ImageNet
- Transfer learning
- Data augmentation
- Global Average Pooling
- Dense classification layers
- Batch Normalization
- Dropout
- Fine-tuning

The input image size is:
224 × 224 × 1

The grayscale input is normalized and converted to RGB before being passed to MobileNetV2.

## Dataset

The dataset is divided into three subsets:

| Split | Images |
|---|---:|
| Training | 18,952 |
| Validation | 2,812 |
| Testing | 1,649 |

Total images:
23,413

The dataset contains five classes:
Cavity
Fillings
Impacted Tooth
Implant
Normal

The dataset is not included in this repository because of its size.

To run the project locally, the dataset must be organized as follows:
dataset/
├── train/
│   ├── Cavity/
│   ├── Fillings/
│   ├── Impacted Tooth/
│   ├── Implant/
│   └── Normal/
│
├── val/
│   ├── Cavity/
│   ├── Fillings/
│   ├── Impacted Tooth/
│   ├── Implant/
│   └── Normal/
│
└── test/
    ├── Cavity/
    ├── Fillings/
    ├── Impacted Tooth/
    ├── Implant/
    └── Normal/

## Data Augmentation

The training pipeline applies:

- Random horizontal flipping
- Random rotation
- Random zoom

## Training

The training process uses transfer learning in two stages.

### Stage 1: Feature Extraction

The MobileNetV2 base model is initially frozen while the classification head is trained.

### Stage 2: Fine-Tuning

The base model is partially unfrozen and the deeper layers are fine-tuned using a lower learning rate.

The training pipeline also uses:

- Early stopping
- Model checkpointing
- Learning-rate reduction on plateau

## Project Structure
Ai-Dental/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
│
├── models/
│   └── best_model.keras
│
└── results/
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── training_curve.png

## Technologies

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

Clone the repository:
git clone https://github.com/shadow-601/Ai-Dental.git

Enter the project directory:
cd Ai-Dental

Install the required dependencies:
pip install -r requirements.txt

## Dataset Setup

Place the dataset directory in the project root:
Ai-Dental/
└── dataset/
    ├── train/
    ├── val/
    └── test/

## Training

To train the model:
python src/train.py

The training process generates the trained model and training results.

## Evaluation

To evaluate the model:
python src/evaluate.py

The evaluation process generates:

- Classification report
- Confusion matrix

## Results

The trained model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 90% |
| Macro F1-Score | 0.66 |
| Weighted F1-Score | 0.90 |

The complete classification report and visualization results are available in the results directory.

The model's performance varies between classes because the dataset is imbalanced, with the Normal class containing significantly more samples than some other categories.

## Pre-trained Model

A trained model is included in:
models/best_model.keras

The model can be loaded using TensorFlow/Keras:
from tensorflow.keras.models import load_model

model = load_model("models/best_model.keras")

## Disclaimer

This project is intended for educational and research purposes.
It is not a certified medical diagnostic system and must not be used as a substitute for professional medical evaluation.

## Author

Ali Saeed Dubai

GitHub:

https://github.com/shadow-601
