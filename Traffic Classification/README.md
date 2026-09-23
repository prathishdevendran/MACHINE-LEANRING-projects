# Urban Traffic Situation Classification

A machine learning classification project designed to categorize urban road traffic intensity into discrete congestion states based on temporal metrics and multi-modal vehicle density distributions.

---

## 1. Project Overview & Problem Statement
Urban traffic congestion leads to economic losses, increased carbon emissions, and transit delays. This project develops a predictive classification framework to determine real-time traffic congestion levels (`low`, `normal`, `high`, `heavy`). By analyzing the composition of vehicle types (cars, bikes, buses, trucks) alongside temporal cycles (time of day, day of week, day of month), the models learn non-linear decision boundaries characterizing traffic bottlenecks and peak-hour flows.

---

## 2. Dataset Architecture
The project utilizes `Traffic.csv` containing granular traffic observation logs:
- **Temporal Indicators**:
  - `Time`: Timestamp of observation in 15-minute interval buckets
  - `Date`: Calendar day of the month (1–31)
  - `Day of the week`: Categorical weekday (Monday through Sunday)
- **Vehicle Count Breakdown**:
  - `CarCount`: Volume of private passenger cars
  - `BikeCount`: Volume of two-wheelers/motorcycles
  - `BusCount`: Volume of public and private transit buses
  - `TruckCount`: Volume of heavy freight and commercial vehicles
  - `Total`: Aggregate vehicle volume across all categories
- **Target Variable (`Traffic Situation`)**:
  - Multi-class congestion label: `low`, `normal`, `high`, `heavy`

---

## 3. Data Processing & Modeling Methodology
The analytical pipeline in `traffic.ipynb` encompasses:

1. **Exploratory Data Analysis (EDA)**:
   - Peak-hour traffic profiling across different days of the week.
   - Vehicle distribution correlation matrices identifying heavy vehicle impact on congestion states.
   - Outlier detection and class distribution analysis.

2. **Feature Engineering & Transformation**:
   - Cyclic and categorical transformation of temporal attributes (Time and Day of Week encoding).
   - Normalization and feature scaling of high-variance vehicle counts.
   - Target label encoding mapping discrete classes to numerical identifiers.

3. **Machine Learning Model Implementations**:
   - **Decision Trees & Random Forest Classifiers**: Capturing non-linear threshold rules governing vehicle density cutoffs.
   - **Gradient Boosting Classifiers (XGBoost / LightGBM / GradientBoosting)**: Sequentially minimizing residual classification errors.
   - **Logistic Regression & K-Nearest Neighbors (KNN)**: Providing baseline linear and distance-based benchmarks.

4. **Model Validation & Metrics**:
   - Stratified K-Fold cross-validation.
   - Multi-class evaluation using Macro/Weighted F1-score, Precision, Recall, and Confusion Matrices.

---

## 4. Key Findings & Insights
- **Vehicle Weighting**: Buses and heavy trucks exert a disproportionate influence on transitioning traffic states from `normal` to `heavy` even at moderate overall volumes.
- **Temporal Peaks**: Weekday morning and evening commuter windows exhibit distinct multimodal distribution spikes compared to weekend traffic dynamics.
- **Tree-Based Superiority**: Ensemble tree methods achieve top accuracy and robust generalization across boundary congestion scenarios.

---

## 5. Execution Instructions

### Environment Setup
Install the necessary computational and visualization libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Running the Analysis
Navigate to the directory and start Jupyter Notebook:
```bash
cd "Traffic Classification"
jupyter notebook traffic.ipynb
```
Execute the notebook cells to reproduce the exploratory plots, data transformations, and model evaluation metrics.
