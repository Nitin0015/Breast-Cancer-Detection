# **Breast Cancer Prediction**

This repository contains a Jupyter Notebook that demonstrates how to classify breast cancer cases as malignant or benign using machine learning techniques. The dataset used in this project is the Breast Cancer Wisconsin Dataset.

---

## **Overview**

Breast cancer is one of the most common cancers affecting women worldwide. Early detection and diagnosis are critical for effective treatment and management. This project uses **Logistic Regression** to classify tumors as malignant or benign based on various features derived from digitized images of fine needle aspirate (FNA) of breast masses.

The dataset includes features such as radius, texture, perimeter, area, and smoothness, with the target variable (`label`) indicating whether the tumor is malignant (`1`) or benign (`0`).

---

## **Dataset**

- **Source**: [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Features**:
  - `mean radius`: Mean of distances from center to points on the perimeter.
  - `mean texture`: Standard deviation of gray-scale values.
  - `mean perimeter`: Mean size of the core tumor.
  - `mean area`: Mean area of the tumor.
  - `mean smoothness`: Local variation in radius lengths.
  - Additional features include compactness, concavity, symmetry, and fractal dimension for both mean and worst-case scenarios.
- **Target Variable**:
  - `label`: Indicates whether the tumor is malignant (`1`) or benign (`0`).

---

## **Project Workflow**

1. **Data Loading**:
   - The dataset is loaded into a Pandas DataFrame.
2. **Exploratory Data Analysis (EDA)**:
   - Summary statistics and visualizations are generated to understand relationships between features and labels.
3. **Data Preprocessing**:
   - Features are scaled to ensure uniformity for better model performance.
4. **Model Training**:
   - A Logistic Regression model is trained to classify tumors.
   - The dataset is split into training and testing sets using `train_test_split`.
5. **Model Evaluation**:
   - Accuracy score and confusion matrix are calculated to evaluate model performance.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

You can install these dependencies using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/BreastCancerPrediction.git
   cd BreastCancerPrediction
   ```

2. Ensure that the dataset file (if external) is in the same directory as the notebook.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Breast-Cancer-Pred.ipynb
   ```

4. Run all cells in the notebook to execute the code.

---

## **Results**

The Logistic Regression model provides predictions for whether a tumor is malignant or benign based on input features. The accuracy score indicates how well the model performs in classifying tumors. Further improvements can be made by experimenting with other machine learning models or feature engineering techniques.

---

## **Acknowledgments**

- The dataset was sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).
- Special thanks to Scikit-learn for providing robust machine learning tools.

---
