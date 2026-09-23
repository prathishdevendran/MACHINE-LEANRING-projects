# Interstate Traffic Forecasting with XGBoost & Grey Wolf Optimizer (GWO)

A research-oriented machine learning time-series regression project combining **Extreme Gradient Boosting (XGBoost)** with the nature-inspired **Grey Wolf Optimization (GWO)** metaheuristic algorithm to forecast hourly interstate highway traffic volume under variable weather conditions.

---

## 1. Project Overview & Scientific Motivation
Accurate long-term and short-term highway traffic volume forecasting is critical for intelligent transportation systems (ITS) and traffic flow management. Standard hyperparameter tuning techniques like Grid Search or Random Search are either computationally prohibitive or prone to sub-optimal local minima. 

This project explores a hybrid bio-inspired optimization framework: using the social leadership hierarchy and hunting mechanism of **Grey Wolves (Alpha, Beta, Delta, and Omega wolves)** to explore and exploit the high-dimensional hyperparameter space of an **XGBoost Regressor**, maximizing predictive accuracy on interstate traffic flows.

---

## 2. Dataset Architecture
The project utilizes `Metro_Interstate_Traffic_Volume.csv`, an extensive hourly time-series dataset (48,200+ records) collected along Interstate 94 (I-94) Westbound in Minneapolis-St. Paul, Minnesota:
- **Target Variable**:
  - `traffic_volume`: Hourly traffic volume counts passing the sensor station.
- **Temporal & Calendar Features**:
  - `date_time`: Hourly timestamp spanning multiple years.
  - `holiday`: US National holidays and regional festive indicators.
- **Meteorological & Atmospheric Features**:
  - `temp`: Average ambient temperature in Kelvin.
  - `rain_1h`: Precipitation amount in mm over the preceding hour.
  - `snow_1h`: Snow accumulation amount in mm over the preceding hour.
  - `clouds_all`: Cloud coverage percentage (0–100%).
  - `weather_main`: High-level weather category (Rain, Snow, Clear, Clouds, Mist, Fog, etc.).
  - `weather_description`: Granular meteorological description text.

---

## 3. Project Notebooks & Methodological Pipeline

The repository is structured into two complementary investigative notebooks:

### 1. `XGBoost_Forecasting.ipynb`
- **Feature Engineering**:
  - Cyclical encoding ($\sin/\cos$) for hour-of-day, day-of-week, and month.
  - Lag features and rolling statistical aggregations (moving averages, exponential smoothing).
  - Categorical encoding for weather states and holiday indicators.
- **Baseline Modeling**:
  - Training standard XGBoost Regressors with default and manual heuristics.
  - Validation using time-based rolling window splits.

### 2. `optimize_GWO.ipynb`
- **Swarm Metaheuristic Implementation**:
  - Encapsulates the GWO algorithm mimicking gray wolf social pack dynamics:
    - $\alpha$ (Alpha): Best-performing hyperparameter solution.
    - $\beta$ (Beta): Second-best hyperparameter solution.
    - $\delta$ (Delta): Third-best hyperparameter solution.
    - $\omega$ (Omega): Remaining candidate solutions following the leadership pack.
- **Hyperparameter Space Optimization**:
  - Optimizes `learning_rate`, `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`, `gamma`, and regularization terms (`reg_alpha`, `reg_lambda`).
  - Iteratively updates wolf positions toward the global minimum fitness function ($RMSE$).

---

## 4. Performance & Evaluation
Models are evaluated using standard regression criteria:
- **Root Mean Squared Error (RMSE)**: Penalizes large traffic prediction deviations.
- **Mean Absolute Error (MAE)**: Average absolute magnitude of error in vehicle counts.
- **$R^2$ (Coefficient of Determination)**: Evaluates variance captured across sudden weather fluctuations and rush-hour spikes.
- **Results**: The GWO-tuned XGBoost model achieves superior convergence speed and lower error rates compared to default baselines and conventional grid search.

---

## 5. Execution Instructions

### Prerequisites
Install the required computational stack:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost jupyter
```

### Running the Notebooks
Navigate to the directory and start Jupyter:
```bash
cd "XGBoost_gwo metro traffic prediction"
jupyter notebook
```
- Run `XGBoost_Forecasting.ipynb` to inspect data preprocessing, baseline model behavior, and time-series feature dynamics.
- Run `optimize_GWO.ipynb` to execute the Grey Wolf Optimization search loop and evaluate the optimized XGBoost architecture.
