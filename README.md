# Dry Bean Classification Using SVM and KNN

## 📌 Project Overview

This project applies two supervised machine learning algorithms, **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)**, to classify different varieties of dry beans.

The project uses the **Dry Bean Dataset** from the UCI Machine Learning Repository. The dataset contains **13,611 samples**, representing **7 different classes of dry beans**. Each sample contains **16 numerical features** extracted from bean images.

The main objective is to train and evaluate SVM and KNN models and compare their classification performance.

---

## 📊 Dataset Information

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Number of Samples:** 13,611  
**Number of Input Features:** 16  
**Target Variable:** `Class`  
**Number of Classes:** 7  

### Bean Classes

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

### Features

The dataset contains the following input features:

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRation
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. Roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

---

## 🎯 Project Objectives

The main objectives of this project are to:

- Explore and understand the Dry Bean Dataset
- Check for missing values and duplicate records
- Analyze the distribution of bean classes
- Prepare the data for machine learning
- Standardize numerical features
- Train an SVM classifier
- Train a KNN classifier
- Evaluate both models
- Compare the performance of SVM and KNN

---

## 🛠️ Technologies and Libraries

The project is implemented using **Python**.

The following libraries are used:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 🔄 Project Workflow

The project follows these steps:

### 1. Data Loading

The Dry Bean Dataset is loaded using Pandas.

### 2. Exploratory Data Analysis

The dataset is examined to understand:

- Dataset dimensions
- Feature names
- Data types
- Missing values
- Duplicate records
- Descriptive statistics
- Target class distribution

### 3. Data Preprocessing

The `Class` column is used as the target variable, while the remaining 16 columns are used as input features.

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

Stratified sampling is used to maintain the original class distribution.

### 4. Feature Scaling

`StandardScaler` is used to standardize the input features.

Feature scaling is particularly important for SVM and KNN because both algorithms are sensitive to differences in feature magnitude.

---

## 🤖 Machine Learning Models

### Support Vector Machine (SVM)

An SVM classifier with the **Radial Basis Function (RBF) kernel** is used.

```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale'
)
