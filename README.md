# Machine Learning Projects Repository

A curated, comprehensive collection of machine learning, deep learning, computer vision, time-series forecasting, and bio-inspired optimization projects. This repository encompasses end-to-end workflows from classical tabular classification and regression to production-ready web services deployed with FastAPI.

---

## 📂 Repository Structure & Project Directory

```text
MACHINE-LEANRING-projects/
├── .gitignore                                 # Git configuration ignoring virtual envs and temp files
├── README.md                                  # Root documentation catalog
│
├── Iris project/                              # Multi-Class Botanical Classification
│   ├── Iris LogisticRegression.ipynb
│   ├── Iris RandomForest.ipynb
│   ├── Iris Svm.ipynb
│   ├── iris.data
│   └── README.md
│
├── ML cat vs dog api/                         # [DEPLOYMENT] FastAPI Deep Learning Web App
│   ├── dogs and cats image/
│   ├── templates/
│   │   └── index.html
│   ├── dogs_vs_cats.h5
│   ├── main.py
│   ├── prediction_model.py
│   ├── requirements.txt
│   └── README.md
│
├── Traffic Classification/                    # Multimodal Urban Traffic Congestion Classification
│   ├── Traffic.csv
│   ├── traffic.ipynb
│   └── README.md
│
├── Traffic-volume-prediction/                 # Big Data Metropolitan Traffic Volume Regression
│   ├── Banglore_traffic_Dataset.csv
│   ├── TrafficVolumePrediction on big data.ipynb
│   └── README.md
│
└── XGBoost_gwo metro traffic prediction/      # Interstate Traffic Forecasting + Grey Wolf Optimization
    ├── Metro_Interstate_Traffic_Volume.csv
    ├── XGBoost_Forecasting.ipynb
    ├── optimize_GWO.ipynb
    └── README.md
```

---

## 📊 Projects Overview & Catalog

| # | Project Name | Domain / Task | Core Algorithms / Frameworks | Target / Output | Documentation |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | [**Iris Project**](file:///e:/projects/MACHINE-LEANRING-projects/Iris%20project/README.md) | Tabular Multi-class Classification | Logistic Regression, Random Forest, SVM (Scikit-Learn) | Iris Flower Species (*Setosa, Versicolor, Virginica*) | [Project README](file:///e:/projects/MACHINE-LEANRING-projects/Iris%20project/README.md) |
| **2** | [**ML Cat vs Dog API**](file:///e:/projects/MACHINE-LEANRING-projects/ML%20cat%20vs%20dog%20api/README.md) *(Deployment)* | Deep Learning Computer Vision & Web Serving | CNN, MobileNetV2, FastAPI, Uvicorn, Jinja2, TensorFlow / tf-keras | Binary Image Classification (`Cat` vs `Dog`) with Confidence % | [Deployment README](file:///e:/projects/MACHINE-LEANRING-projects/ML%20cat%20vs%20dog%20api/README.md) |
| **3** | [**Traffic Classification**](file:///e:/projects/MACHINE-LEANRING-projects/Traffic%20Classification/README.md) | Multimodal Traffic Density Classification | Decision Trees, Random Forest, Gradient Boosting | Congestion Severity Level (`low`, `normal`, `high`, `heavy`) | [Project README](file:///e:/projects/MACHINE-LEANRING-projects/Traffic%20Classification/README.md) |
| **4** | [**Traffic Volume Prediction**](file:///e:/projects/MACHINE-LEANRING-projects/Traffic-volume-prediction/README.md) | Big Data Urban Mobility Regression | Ridge/Lasso, Random Forest, XGBoost Regressor | Continuous Intersection Traffic Volume & Congestion Index | [Project README](file:///e:/projects/MACHINE-LEANRING-projects/Traffic-volume-prediction/README.md) |
| **5** | [**XGBoost + GWO Metro Traffic**](file:///e:/projects/MACHINE-LEANRING-projects/XGBoost_gwo%20metro%20traffic%20prediction/README.md) | Time-Series Forecasting & Metaheuristic Optimization | XGBoost Regressor + Grey Wolf Optimizer (GWO Swarm Intelligence) | Hourly Interstate-94 Highway Traffic Volume | [Project README](file:///e:/projects/MACHINE-LEANRING-projects/XGBoost_gwo%20metro%20traffic%20prediction/README.md) |

---

## 🔍 Detailed Project Descriptions

### 1. [Iris Flower Classification](file:///e:/projects/MACHINE-LEANRING-projects/Iris%20project/README.md)
- **Objective**: Benchmark three distinct machine learning paradigms on the classic 150-sample Fisher's Iris botanical dataset.
- **Methodology**: Evaluates linear probabilistic decision surfaces (Logistic Regression), non-linear margin-maximizing hyperplanes with kernel projections (Support Vector Machines), and ensemble decision tree bagging (Random Forest).
- **Key Takeaways**: Demonstrates feature separability dynamics (Petal Length and Petal Width acting as principal discriminators) and provides a template for cross-model comparative evaluation.

### 2. [Cat vs Dog Deep Learning Web API (Deployment)](file:///e:/projects/MACHINE-LEANRING-projects/ML%20cat%20vs%20dog%20api/README.md)
- **Objective**: Full-stack machine learning deployment serving a deep convolutional neural network for real-time binary image classification.
- **Architecture**:
  - **Serving Layer**: FastAPI backend with async request handling and Swagger/OpenAPI documentation.
  - **Inference Engine**: Robust TensorFlow / `tf-keras` loader with automatic layer fallback (`CustomDepthwiseConv2D`) and in-memory PIL image transformation.
  - **Frontend UI**: Jinja2-rendered interactive web page supporting file upload, base64 image preview, and animated classification badges.
- **Execution Guide**: Refer to the dedicated [ML Cat vs Dog API README](file:///e:/projects/MACHINE-LEANRING-projects/ML%20cat%20vs%20dog%20api/README.md) for step-by-step local setup, virtual environment configuration, and server launching commands.

### 3. [Urban Traffic Situation Classification](file:///e:/projects/MACHINE-LEANRING-projects/Traffic%20Classification/README.md)
- **Objective**: Classify urban roadway congestion states into four discrete levels (`low`, `normal`, `high`, `heavy`).
- **Data & Features**: Processes 15-minute interval records encompassing vehicle modality counts (passenger cars, motorcycles, buses, heavy trucks) and temporal calendar indicators.
- **Insights**: Heavy commercial vehicles and buses exert a non-linear compounding impact on bottleneck formation compared to passenger vehicles.

### 4. [Bangalore Big Data Traffic Volume & Congestion Prediction](file:///e:/projects/MACHINE-LEANRING-projects/Traffic-volume-prediction/README.md)
- **Objective**: Model complex metropolitan traffic flow and road capacity saturation across major arterial corridors and intersections in Bangalore.
- **Data & Features**: Multi-variable spatial dataset containing vehicle speeds, travel time index (TTI), road capacity utilization, weather conditions, accident incident reports, and public transit adoption.
- **Modeling**: Employs ensemble tree regressors and gradient boosting architectures to forecast continuous traffic volume for intelligent city routing.

### 5. [Interstate Traffic Forecasting with XGBoost & Grey Wolf Optimization](file:///e:/projects/MACHINE-LEANRING-projects/XGBoost_gwo%20metro%20traffic%20prediction/README.md)
- **Objective**: Accurately forecast hourly interstate highway traffic volume on I-94 Westbound under extreme weather variations using bio-inspired metaheuristic search.
- **Methodology**: Combines Extreme Gradient Boosting (XGBoost) with Grey Wolf Optimizer (GWO). GWO simulates the predatory hunting hierarchy ($\alpha$, $\beta$, $\delta$, $\omega$ wolves) to dynamically search and fine-tune continuous and discrete XGBoost hyperparameters, outperforming standard grid/random search.

---

## 🛠️ Repository Setup & Guidelines

### Virtual Environments & Git Tracking
- All virtual environment directories (e.g. `ml-fastapi-env/`, `venv/`, `.venv/`), Python cache files (`__pycache__/`), and Jupyter checkpoints are ignored via the root [`.gitignore`](file:///e:/projects/MACHINE-LEANRING-projects/.gitignore).
- All source code, datasets (`.csv`, `.data`), pretrained weights (`.h5`), and Jupyter notebooks (`.ipynb`) are fully preserved and tracked.

### Global Dependencies
To run all notebook projects locally, install the foundational scientific stack:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost jupyter
```
For the web deployment service, follow the dedicated instructions in [ML cat vs dog api/README.md](file:///e:/projects/MACHINE-LEANRING-projects/ML%20cat%20vs%20dog%20api/README.md).
