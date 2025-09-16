# 🫀 Heart Disease Predictor

## 📌 Project Overview

This project applies machine learning algorithms to predict whether a person has heart disease based on their medical and personal records.
The goal is to explore the dataset, perform feature engineering, and compare the performance of multiple ML models for classification.

Dataset used: [Kaggle – Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/endofnight17j03/heart-failure-prediction-dataset)

## ⚙️ Workflow

### 1. Exploratory Data Analysis (EDA):

Summary statistics, correlations, and feature relationships

### 2. Data visualization 

Visualizing variables and correlations using heatmaps, histograms, scatterplots, box plots, and more

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/28de72fd-a3fe-4911-86bf-c3d5ee407069" />

<img width="845" height="553" alt="image" src="https://github.com/user-attachments/assets/19794984-2ab1-4047-b203-a387b85ccecf" />

<img width="540" height="393" alt="image" src="https://github.com/user-attachments/assets/3e776ca1-4fc3-4aaf-b1de-372d39ff6e65" />

### 3. Data Preprocessing

Removal of duplicate rows, handling of missing values, and encoding categorical variables
Standardization and dimensionality reduction using PCA
Train/Test split

### 4. Machine Learning
Implemented six algorithms:

- K-Nearest Neighbors (KNN)

- Naive Bayes

- Decision Tree

- Random Forest

- Support Vector Machine (SVM)

- Multilayer Perceptron (MLP, TensorFlow)

### 4. Model Evaluation

Accuracy, Recall, and Precision

#### 📊 Results
| Model            | Accuracy  | Recall    | Precision |
| ---------------- | --------- | --------- | --------- |
| **KNN**          | **85.3%** | **88.8%** | 86.4%     |
| Naive Bayes      | 84.2%     | 86.9%     | 86.1%     |
| Decision Tree    | 75.5%     | 77.6%     | 79.8%     |
| Random Forest    | 84.2%     | 86.9%     | 86.1%     |
| SVM              | 84.8%     | 86.9%     | 86.9%     |
| MLP (TensorFlow) | 82.1%     | 81.3%     | **87.0%** |

✅ Best overall performance: KNN with 85.3% accuracy and 88.8% recall.

## 🛠️ Tech Stack
- Python
- Libraries: Pandas, NumPy, Scikit-learn, TensorFlow, Seaborn, Matplotlib

## 🚀 How to Run
1. Clone the repository:
```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:
```bash
jupyter notebook heart_disease_predictor.ipynb
```
