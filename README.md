<div align="center">

<img src="heart_ml_banner.png" alt="Heart & Machine Learning - Advancing Cardiovascular Care with AI" width="100%"/>

# ❤️ Heart Disease Prediction using Machine Learning

**An end-to-end ML pipeline that predicts the presence of heart disease from patient medical attributes.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

</div>

---

## 📌 Project Objective

Heart disease is one of the leading causes of death worldwide. Early prediction can assist healthcare professionals in identifying high-risk patients and taking preventive measures.

This project performs a complete, end-to-end machine learning workflow to predict the presence of heart disease using patient medical attributes, covering everything from raw data to a deployable classification model.

**The pipeline includes:**

- 🔍 Data Exploration & Cleaning
- 📊 Exploratory Data Analysis (EDA)
- 🔗 Correlation Analysis
- 🎯 Feature Selection (Pearson Correlation + ANOVA F-Test)
- ⚙️ Data Preprocessing & Scaling
- 🤖 Model Training (5 algorithms)
- 📈 Stratified Cross-Validation
- 🛠️ Hyperparameter Tuning (GridSearchCV)
- 🧪 Model Evaluation & Interpretability
- 🔥 Final Prediction Demo

---

## 🗂️ Dataset

| | |
|---|---|
| **Samples** | 270 patient records |
| **Features** | 13 clinical attributes |
| **Target** | `Heart Disease` — Presence / Absence |
| **File** | `Heart_Disease_Prediction.csv` |

<details>
<summary><b>📋 Full feature list</b></summary>

| Feature | Description |
|---|---|
| `Age` | Patient age (years) |
| `Gender` | 0 = Female, 1 = Male |
| `Chest pain type` | Type of chest pain experienced |
| `BP` | Resting blood pressure |
| `Cholesterol` | Serum cholesterol level |
| `FBS over 120` | Fasting blood sugar > 120 mg/dl |
| `EKG results` | Resting electrocardiographic results |
| `Max HR` | Maximum heart rate achieved |
| `Exercise angina` | Exercise-induced angina (0 = No, 1 = Yes) |
| `ST depression` | ST depression induced by exercise |
| `Slope of ST` | Slope of the peak exercise ST segment |
| `Number of vessels fluro` | Number of major vessels colored by fluoroscopy |
| `Thallium` | Thallium stress test result |
| `Heart Disease` | **Target** — Presence (1) / Absence (0) |

</details>

---

## 🧠 Methodology

### 1. Data Cleaning
- Target column mapped from categorical strings (`Presence` / `Absence`) to binary (`1` / `0`)
- All columns cast to a consistent numeric type
- No missing values found in the dataset

### 2. Exploratory Data Analysis
Visualized gender distribution, age distribution, age vs. cholesterol trends, outliers across clinical features (via boxplots), feature distributions, and a correlation heatmap to understand relationships in the data.

### 3. Feature Selection
Two statistical techniques were used to identify the most predictive features:

- **Pearson Correlation** — measures linear correlation of each feature with the target
- **ANOVA F-Test** (`SelectKBest`) — ranks features by statistical significance (p-value)

**Top features selected for modeling** (ranked by F-score):

| Rank | Feature | F-Score | p-value |
|---|---|---|---|
| 1 | Thallium | 101.99 | 1.57e-20 |
| 2 | Number of vessels fluro | 70.10 | 3.17e-15 |
| 3 | Exercise angina | 57.17 | 6.38e-13 |
| 4 | Chest pain type | 56.55 | 8.26e-13 |
| 5 | ST depression | 54.56 | 1.91e-12 |
| 6 | Slope of ST | 34.48 | 1.27e-08 |
| 7 | Gender | 26.07 | 6.27e-07 |
| 8 | Age | 12.65 | 4.43e-04 |

➡️ **Final feature set (8 features):** `Thallium`, `Exercise angina`, `BP`, `Number of vessels fluro`, `Chest pain type`, `Age`, `ST depression`, `Gender`

### 4. Preprocessing
- Train/Test split: **80% / 20%** (`random_state=42`)
- Feature scaling via `StandardScaler` (fit on train, applied to test)

### 5. Model Training & Cross-Validation
Five classifiers were trained and evaluated using **5-fold Stratified Cross-Validation**:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier (SVC)
- Random Forest
- XGBoost

### 6. Hyperparameter Tuning
`GridSearchCV` was used to tune Logistic Regression over a grid of `C`, `penalty`, and `solver` values.

---

## 📊 Results

### Cross-Validation Performance (Train Set)

| Model | Mean Accuracy | Std Dev |
|---|---|---|
| **SVC** | **83.32%** | ±4.78% |
| Random Forest | 82.84% | ±7.34% |
| **Logistic Regression** | **82.42%** | ±4.25% |
| KNN | 80.11% | ±5.16% |
| XGBoost | 78.25% | ±9.36% |

> SVC and Logistic Regression were the top performers. Random Forest and XGBoost showed more variance, likely due to the small dataset size (270 samples), where simpler linear/margin-based models tend to generalize better.

### Final Model: Tuned Logistic Regression

**Best hyperparameters:** `{'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}`

**Test Set Performance (54 held-out samples):**

| Metric | No Disease (0) | Disease (1) | Overall |
|---|---|---|---|
| Precision | 0.84 | 0.88 | — |
| Recall | 0.94 | 0.71 | — |
| F1-score | 0.89 | 0.79 | — |
| **Accuracy** | | | **85.19%** |

**Confusion Matrix:**

|  | Predicted: No Disease | Predicted: Disease |
|---|---|---|
| **Actual: No Disease** | 31 | 2 |
| **Actual: Disease** | 6 | 15 |

### Why Logistic Regression?

- ✅ **Interpretability** — coefficients map directly to clinical impact, valuable in a healthcare context
- ✅ **Strong, stable performance** across cross-validation folds
- ✅ **L1 regularization** naturally performs feature selection, reducing overfitting risk on a small dataset
- ✅ Comparable accuracy to SVC, with far greater transparency for clinical decision-making

---

## 🔥 Sample Prediction

The trained model was tested on two synthetic patient profiles:

```text
Healthy Profile   → Prediction: Absence of Heart Disease
Unhealthy Profile → Prediction: Presence of Heart Disease
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, XGBoost |
| **Statistics** | SciPy |

---

## 📁 Project Structure

```
heart-disease-prediction/
├── heart.ipynb                     # Main notebook — full ML pipeline
├── Heart_Disease_Prediction.csv    # Dataset
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost scipy
```

### Run the Project

```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
jupyter notebook heart.ipynb
```

---

## 🔮 Future Improvements

- Expand the dataset to improve model robustness and reduce variance across folds
- Experiment with ensemble/stacking approaches combining Logistic Regression and SVC
- Deploy the model as a REST API or interactive web app (e.g., Streamlit/FastAPI)
- Add SHAP-based explainability for individual predictions
- Perform more extensive hyperparameter tuning across all candidate models

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It is **not** a certified medical diagnostic tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ and Machine Learning**

</div>
