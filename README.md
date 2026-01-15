# Computer Vision Project: CNN Classifier & Scene Recognition

**Author:** Davide Zorz  
**Date:** January 2025

## 📋 Project Overview

This project implements various image classification techniques on a scene recognition dataset containing 15 categories (e.g., Office, Kitchen, Living Room, Coast, etc.). The project progresses from building custom Convolutional Neural Networks (CNNs) from scratch to leveraging Transfer Learning with AlexNet and implementing Multiclass SVMs.

The solution features a custom **Parser** and **Dynamic Model Builder** that allows defining network architectures via configuration files without changing Python code.

## 📂 Project Structure

The project is organized as follows:

- **`solve.py`**: The main solver script for training and evaluating standard CNN models.
- **`point2f.py`**: Specialized solver for the Ensemble method (Point 2f).
- **`point3a.py`**: Solver for AlexNet Fine-tuning.
- **`pointSVM.py`**: Solver for SVM-based classification (Linear, RBF, ECOC).
- **`configs/`**: Contains `.yaml` files defining hyperparameters, architecture, and data pipelines for each exercise.
- **`models/`**:
  - `generalCNN.py`: Dynamic CNN builder.
  - `SVMs.py`: Implementations of Multiclass SVMs (OVO, DAG, ECOC).
  - `ViT.py`: Implementation of a vision transformer.
- **`dataset/`**:
  - `dataloader.py`: Handles image loading, stratified splitting, and transformations.
- **`parsing/`**:
  - `convParser.py`: Parses string configurations into PyTorch modules.

## ⚙️ Requirements

* Python 3.x
* PyTorch
* Torchvision
* Scikit-learn
* NumPy
* PyYAML
* Matplotlib

Before running the scripts, ensure you have the necessary dependencies installed. 
```bash
pip install -r requirements.txt
```


## 💾 Dataset Preparation

The system expects the dataset to be organized into training and testing folders. The `DatasetManager` in `dataset/dataloader.py` handles the loading.

⚠️ Important: The scripts assume your data is organized in a ```.data/``` directory. Ensure your file structure matches the following tree exactly before running the training script:

```
.data/
  ├ train/
  │  ├ Category_1/
  │  │  ├ image_1.jpg
  │  │  ├ ...
  │  ├ Category_2/
  │  │  ├ image___.jpg
  │  │  ├ ...
  │  ├ .../
  │  │  ├ ...
  │  │  ├ ...
  │  └ Category_n/
  │     ├ image___.jpg
  │     └ ...
  └ test/
    ├ ... 
    │  ├ ... 
    │  └ ...
    ├ ... 
    ...
```

## 🚀 Usage & Commands

Below are the commands to reproduce the results for every point in the report.




### Part 1: Custom CNNs (From Scratch)

**Point 1: Basic CNN**
A simple 3-block CNN.
```bash
python solve.py --config-path configs/point1.yaml
```

**Point 2: Architecture & Pipeline Improvements**
* **2a (Data Augmentation):** Random horizontal flipping.
    ```bash
    python solve.py --config-path configs/point2a.yaml
    ```
* **2b (Batch Normalization):** Added BatchNorm before ReLU.
    ```bash
    python solve.py --config-path configs/point2b.yaml
    ```
* **2c (Deeper Network):** Larger blocks to learn complex features.
    ```bash
    python solve.py --config-path configs/point2c.yaml
    ```
* **2d (Weight Init & Optimizer):** Kaiming Initialization and AdamW optimizer.
    ```bash
    python solve.py --config-path configs/point2d.yaml
    ```
* **2e (Dropout):** Added Dropout for regularization.
    ```bash
    python solve.py --config-path configs/point2e.yaml
    ```

**Point 2f: Ensemble Method**
An ensemble of 9 networks (same configuration as 2e).
```bash
python point2f.py --config-path configs/point2f.yaml
```

**Point 4: Stronger Augmentation**
Network from 2e trained with rotation, cropping, and flipping.
```bash
python solve.py --config-path configs/point4b.yaml
```

**Point 5: Large Custom CNN**
A deeper custom architecture with 4 blocks and increasing filters (up to 64).
```bash
python solve.py --config-path configs/point5a.yaml
```

---

### Part 2: Transfer Learning (AlexNet)

**Point 3a: Fine-Tuning**
Replaces the last layer of a pretrained AlexNet and fine-tunes it on the dataset.
```bash
python point3a.py --config-path configs/point3a.yaml
```

---

### Part 3: Support Vector Machines (SVM)

These scripts use a pretrained AlexNet as a feature extractor (penultimate layer, 4096 features) and train SVMs on the extracted features.

**Point 3b: Linear SVM**
Implements One-vs-One (OVO) and Directed Acyclic Graph (DAG) strategies.
```bash
python pointSVM.py --config-path configs/point3b.yaml
```

**Point 6: RBF Kernel SVM**
SVM using a Radial Basis Function (Gaussian) kernel.
```bash
python pointSVM.py --config-path configs/point6.yaml
# For OVO specific configuration:
python pointSVM.py --config-path configs/point6ovo.yaml
```

**Point 7: Error Correcting Output Codes (ECOC)**
Multiclass classification using binary codewords (Randomized Hill Climbing generation).
```bash
python pointSVM.py --config-path configs/point7.yaml
# For RBF kernel with ECOC:
python pointSVM.py --config-path configs/point7rbf.yaml
```

## 📊 Results Summary

| Method | Description | Test Accuracy (Approx) |
| :--- | :--- | :--- |
| **Point 1** | Basic CNN | ~25.2% |
| **Point 2a** | + Data Augmentation | ~33.2% |
| **Point 2b** | + Batch Normalization | ~50.2% |
| **Point 2d** | + Kaiming Init & AdamW | ~57.0% |
| **Point 2e** | + Dropout | ~59.4% |
| **Point 2f** | **Ensemble (9 Networks)** | **~63.1%** |
| **Point 5** | Large Custom CNN | ~64.9% |
| **Point 3a** | **AlexNet Fine-Tuning** | **~85.2%** |
| **Point 3b** | AlexNet + Linear SVM | ~85.3% |
| **Point 6** | AlexNet + RBF SVM | ~86.3% |
| **Point ViT** | ViT | ~38.86% |

## 🛠 Technical Highlights

### Custom Parser (`parsing/convParser.py`)
To simplify architecture definitions, a string-based parser was implemented. It converts a list of strings in the YAML config into a PyTorch `nn.Sequential` model.
* **Example Input:** `['conv2d', 'channels: 8', 'kernel_size: (3,3)', 'relu', 'maxpool2d']`
* **Functionality:** Automatically calculates output dimensions between blocks (handling Flatten layers).

### Training Loop (`train.py`)
Features a robust training procedure including:
* **Early Stopping:** Monitors validation loss to prevent overfitting.
* **Best Model Reloading:** Automatically restores weights from the epoch with the lowest validation loss.
* **Schedulers:** Uses `LinearLR` for learning rate decay.




## Side Comments

─  │
┌ ┐ ╭ ╮
└ ┘ ╰ ╯
┬  ┴  ├  ┤
┼

◀

