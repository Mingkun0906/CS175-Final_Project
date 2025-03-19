# CS175-Final_Project: Facial Expression Recognition

## Overview

This project implements three different deep learning models (CNN, MLP, and ResNet) for facial expression recognition. The models are trained using PyTorch and evaluated on a dataset of facial expressions.

---

## Features

- CNN, MLP, and ResNet models for facial expression classification.
- Data preprocessing and augmentation techniques.
- Training and testing scripts with model evaluation.
- Model saving and reloading functionality.
- Accuracy results stored in Excel files for analysis.

---

### Requirements

- **Python 3.8+**
- **Torch & Torchvision** (for deep learning models)
- **Pandas** (for storing accuracy results)
- **Matplotlib** (for visualizing performance metrics)
- **OpenPyXL** (for saving accuracy results to Excel)

---

## Setup Guide

### Install Dependencies
#### Install dependencies:
```
pip install torch torchvision pandas matplotlib openpyxl
```

#### If using a GPU, install PyTorch with CUDA support:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Prepare Dataset
Ensure your dataset is stored in emotion_origins/ with the following structure:
```
emotion_origins/
    class_1/  # Example: happy
        img_1.jpg
        img_2.jpg
    class_2/  # Example: sad
        img_3.jpg
        img_4.jpg
```

## Run Code
#### Train and Test CNN
```
python3 cnn_train.py
```
This will save the trained model as conv_cnn_model.pth and accuracy results in conv_accuracies.xlsx.

#### Train and Test MLP
```
python3 mlp_train.py
```
This will save the trained model as mlp_model.pth and accuracy results in mlp_accuracies.xlsx.

#### Train and Test ResNet50
```
python3 resnet_train.py
```
This will save the trained model as resnet50_model.pth and accuracy results in resnet50_accuracies.xlsx.

## Load and Test Pretrained Models
After training, you can reload the models and test them using:

#### Test CNN Model
```
python3 test_model.py --model conv_cnn_model.pth --type cnn
```
#### Test MLP Model
```
python3 test_model.py --model mlp_model.pth --type mlp
```
#### Test ResNet50 Model
```
python3 test_model.py --model resnet50_model.pth --type resnet
```
These commands will load the saved models and evaluate them on the test dataset.

## Results
Each model generates accuracy results stored in an Excel file:
- conv_accuracies.xlsx (CNN results)
- mlp_accuracies.xlsx (MLP results)
- resnet50_accuracies.xlsx (ResNet results)
Use these files to analyze and compare model performance.

## Notes
- Training on a CPU may take significantly longer than on a GPU.
