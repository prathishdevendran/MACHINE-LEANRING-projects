# Iris Flower Classification Project

An end-to-end multi-class classification machine learning study evaluating classical algorithmic paradigms on the canonical Fisher's Iris dataset.

---

## 1. Project Overview & Objectives
The primary objective of this project is to benchmark and compare fundamental supervised machine learning classification algorithms on a well-understood, multi-class botanical dataset. The project explores feature separability, decision boundary geometry, and predictive performance across distinct algorithmic families: linear probabilistic modeling, margin-maximization hyperplanes, and ensemble decision tree architectures.

---

## 2. Dataset Architecture
The study utilizes the classic Fisher's Iris Dataset (`iris.data`), which captures morphological variations across three related species of Iris flowers:
- **Total Samples**: 150 instances (50 samples per class, perfectly balanced)
- **Input Features (4 Continuous Dimensions in cm)**:
  - `Sepal Length`: Longitudinal measurement of the outer flower calyx
  - `Sepal Width`: Lateral measurement of the outer flower calyx
  - `Petal Length`: Longitudinal measurement of the inner flower corolla
  - `Petal Width`: Lateral measurement of the inner flower corolla
- **Target Classes (Categorical Multi-class)**:
  - `Iris-setosa`: Linearly separable from the other two species
  - `Iris-versicolor`: Partially overlapping with Virginica in feature space
  - `Iris-virginica`: Partially overlapping with Versicolor in feature space

---

## 3. Implemented Models & Notebook Breakdown
The repository contains three dedicated Jupyter notebooks exploring different machine learning methodologies:

1. **`Iris LogisticRegression.ipynb`**:
   - Implements Multinomial / One-vs-Rest (OvR) Logistic Regression.
   - Applies sigmoid/softmax activation functions to map linear feature combinations into class probabilities.
   - Analyzes feature weights to determine which measurements contribute most strongly to species classification.

2. **`Iris RandomForest.ipynb`**:
   - Implements an ensemble Random Forest Classifier aggregating multiple de-correlated decision trees.
   - Leverages bootstrap aggregation (bagging) and random feature subspace selection to minimize variance and combat overfitting.
   - Extracts Gini-impurity-based feature importance scores, highlighting petal dimensions as the primary discriminators.

3. **`Iris Svm.ipynb`**:
   - Implements Support Vector Machine (SVM) classification.
   - Analyzes optimal hyperplane separation using both Linear and Radial Basis Function (RBF) kernels to project non-linear boundaries into higher-dimensional feature spaces.
   - Evaluates margin maximization and support vector density.

---

## 4. Workflow & Evaluation Pipeline
Each notebook adheres to a standardized machine learning workflow:
- **Exploratory Data Analysis (EDA)**: Pair plots, distribution plots, and correlation matrices to inspect multi-collinearity.
- **Data Preprocessing**: Feature scaling (StandardScaler/MinMaxScaler) and label encoding of botanical classes.
- **Train-Test Splitting**: Stratified 80/20 train-test splits preserving balanced class proportions.
- **Model Evaluation Metrics**:
  - Classification Accuracy Score
  - Precision, Recall, and F1-Score per class
  - Confusion Matrix visualization to identify misclassification clusters between *Iris-versicolor* and *Iris-virginica*.

---

## 5. Getting Started & Execution

### Prerequisites
Ensure you have Python 3.8+ installed along with standard data science libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Running the Notebooks
Launch the Jupyter Notebook environment from this project directory:
```bash
jupyter notebook
```
Open any of the three analysis notebooks (`Iris LogisticRegression.ipynb`, `Iris RandomForest.ipynb`, or `Iris Svm.ipynb`) and execute all cells sequentially (`Kernel` -> `Restart & Run All`).
