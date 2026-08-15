# 🛡️ Insurance Claim & Fraud Prediction

> An end-to-end Machine Learning pipeline for insurance risk analysis,
> claim prediction and fraud detection using statistical modeling and
> Random Forest.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

------------------------------------------------------------------------

## 📌 Overview

This project implements a complete and reproducible Machine Learning
pipeline for insurance risk modeling.

The pipeline processes insurance data, performs preprocessing and
feature engineering, compares a Logistic Regression baseline with an
optimized Random Forest classifier, and evaluates model performance
using metrics designed for imbalanced classification.

The project is designed with a modular architecture so that different
insurance datasets can be integrated without rewriting the entire
pipeline.

------------------------------------------------------------------------

## 🎯 Objectives

-   Build a reproducible Machine Learning pipeline for insurance data
-   Compare Logistic Regression with Random Forest
-   Handle imbalanced binary classification
-   Optimize Random Forest hyperparameters
-   Optimize the classification decision threshold
-   Evaluate model performance using robust metrics
-   Generate predictions and reusable model artifacts
-   Provide interpretable feature importance

------------------------------------------------------------------------

## 🧠 Machine Learning Approach

### Logistic Regression

Logistic Regression is used as a baseline statistical model.

It provides an interpretable reference against which the Random Forest
model can be evaluated.

Class imbalance is addressed using:

``` python
class_weight="balanced"
```

### Random Forest

The main Machine Learning model is a Random Forest classifier.

Hyperparameters are optimized using:

``` text
RandomizedSearchCV
```

Configuration:

-   30 randomized parameter combinations
-   5-fold stratified cross-validation
-   F1-score optimization

------------------------------------------------------------------------

## ⚖️ Handling Class Imbalance

Insurance classification problems can contain significantly fewer
positive observations than negative observations.

Using the default probability threshold of `0.50` can therefore lead to
poor minority-class detection.

This project evaluates the Precision-Recall relationship and selects a
decision threshold designed to maximize the F1-score.

  Metric      Purpose
  ----------- --------------------------------------
  Accuracy    Overall classification performance
  Precision   Reliability of positive predictions
  Recall      Ability to detect positive cases
  F1-score    Balance between precision and recall
  ROC-AUC     Model discrimination ability

------------------------------------------------------------------------

## 📊 Dataset

The project supports the publicly available Insurance Claims Fraud
Dataset by Derrick Mwiti.

Dataset: https://github.com/mwitiderrick/insurancedata

The dataset contains approximately 1,000 automobile insurance claims
with a binary `fraud_reported` target. Approximately 25% of observations
are labeled as fraudulent.

### Important modeling consideration

The public dataset contains claims that have already occurred.

Therefore, with this dataset, the prediction target is:

> **Is the reported claim fraudulent?**

rather than:

> **Will a policyholder experience a claim?**

Claim-occurrence prediction requires policy-level observations
containing both claims and non-claims.

------------------------------------------------------------------------

## 🔄 Target Modes

### Fraud Detection

``` python
TARGET_MODE = "label"
```

Target: `fraud_reported`

This mode predicts whether an existing claim is fraudulent.

### Claim Occurrence

``` python
TARGET_MODE = "amount_threshold"
```

Target:

``` text
SINISTRE_CIBLE = 1 if SINISTRE_REGLE > 0
```

This mode can be used with a policy-level dataset containing both claims
and non-claims.

------------------------------------------------------------------------

## 🗂️ Feature Mapping

Dataset-specific mappings are centralized in `src/config.py`.

  -------------------------------------------------------------------------
  Source Column             Internal Feature        Description
  ------------------------- ----------------------- -----------------------
  `age`                     `AGE_ASSURE`            Age of the insured

  `months_as_customer`      `DUREE_MOIS_CONTRAT`    Customer / contract
                                                    duration

  `policy_annual_premium`   `PRIME`                 Annual insurance
                                                    premium

  `umbrella_limit`          `CAPITAL_ASSURE`        Insured capital

  `total_claim_amount`      `SINISTRE_REGLE`        Claim amount

  `fraud_reported`          Target                  Fraud label
  -------------------------------------------------------------------------

To adapt the pipeline to another dataset, modify `RENAME_MAP` and
`TARGET_MODE` in `src/config.py`.

------------------------------------------------------------------------

## 🏗️ Project Structure

``` text
insurance-claim-prediction/
│
├── data/
│   └── donnees_sinistres.csv
│
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── main.py
│
├── results/
│   ├── random_forest_sinistres.png
│   ├── predictions_random_forest.csv
│   ├── importance_variables.csv
│   └── random_forest_model.joblib
│
├── notebooks/
│   └── analyse_sinistres.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## 🔬 Methodology

``` text
Raw Insurance Data
        │
        ▼
Data Loading & Validation
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Stratified Validation
        │
        ├──────────────────┐
        ▼                  ▼
Logistic Regression   Random Forest
Baseline              + Hyperparameter Search
        │                  │
        └────────┬─────────┘
                 ▼
          Cross-Validation
                 │
                 ▼
          Model Evaluation
                 │
                 ▼
      Precision-Recall Analysis
                 │
                 ▼
      Decision Threshold Selection
                 │
                 ▼
       Predictions & Artifacts
```

### Data preprocessing

The pipeline performs:

-   Column normalization
-   Data type validation
-   Date parsing when applicable
-   Missing-value handling
-   Outlier detection
-   Feature engineering
-   Feature selection

### Model training

A balanced Logistic Regression model establishes a statistical
benchmark. Random Forest hyperparameters are optimized using
`RandomizedSearchCV` with stratified 5-fold cross-validation and
F1-score optimization.

### Decision threshold

Instead of automatically using `0.50`, the pipeline evaluates
probability thresholds using Precision-Recall analysis. This is
particularly relevant for imbalanced classification.

### Model interpretation

Random Forest feature importance is extracted to identify the variables
contributing most strongly to model predictions.

------------------------------------------------------------------------

## 📈 Results & Interpretation

With the current configuration using only four numerical features:

-   Age
-   Insured capital
-   Premium
-   Contract duration

the ROC-AUC obtained for fraud prediction is close to `0.50`.

This indicates that these variables alone provide limited information
for distinguishing fraudulent from non-fraudulent claims.

This result is expected because fraud is strongly influenced by
contextual information surrounding the incident.

Potentially useful features include:

-   Incident severity
-   Type of collision
-   Number of witnesses
-   Police report availability
-   Accident location
-   Vehicle characteristics
-   Previous claims
-   Customer history
-   Policy characteristics

> **Model performance depends not only on the algorithm, but also on the
> quality and predictive power of the available features.**

------------------------------------------------------------------------

## 📊 Generated Outputs

### Trained model

`random_forest_model.joblib` --- reusable trained Random Forest model.

### Predictions

`predictions_random_forest.csv` --- predicted classes and probabilities.

### Feature importance

`importance_variables.csv` --- ranking of the variables used by the
model.

### Visualization

`random_forest_sinistres.png` --- model evaluation visualizations
including confusion matrix, ROC curve, feature importance and
performance metrics.

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repository:

``` bash
git clone <YOUR_REPOSITORY_URL>
cd insurance-claim-prediction
```

Create a virtual environment:

``` bash
python3 -m venv venv
```

macOS / Linux:

``` bash
source venv/bin/activate
```

Windows:

``` bash
venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Usage

Place the dataset inside:

``` text
data/donnees_sinistres.csv
```

Run the complete pipeline:

``` bash
python -m src.main
```

Fast mode without hyperparameter optimization:

``` bash
python -m src.main --skip-search
```

Custom dataset:

``` bash
python -m src.main --data data/my_dataset.csv
```

------------------------------------------------------------------------

## 🔧 Configuration

Most pipeline parameters are centralized in `src/config.py`, including:

-   Feature mapping
-   Target definition
-   Candidate features
-   Random Forest parameters
-   Cross-validation settings
-   Hyperparameter search
-   Decision threshold strategy
-   Output paths

This design makes the pipeline easier to maintain, test and adapt to new
insurance datasets.

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Add richer insurance and incident features
-   Evaluate XGBoost, LightGBM and Gradient Boosting
-   Add SHAP for Explainable AI
-   Expose the model through FastAPI
-   Build a Streamlit prediction application
-   Add Docker and CI/CD
-   Add model monitoring
-   Deploy the application to the cloud

------------------------------------------------------------------------

## ⚠️ Limitations

-   The current feature set contains a limited number of numerical
    variables.
-   The public dataset contains claims that have already occurred.
-   Fraud detection is different from claim-occurrence prediction.
-   Median imputation is currently applied globally.
-   The model is not yet exposed through a production API.
-   The current features are insufficient for strong fraud
    discrimination.

These limitations are explicitly documented to ensure transparent and
reproducible Machine Learning experimentation.

------------------------------------------------------------------------

## 👨‍💻 Author

**Fleury Niyokwizera**

🎓 Master's Student in Applied Mathematics & Statistics\
🤖 Aspiring AI Engineer\
📊 Machine Learning · Data Science · Statistical Modeling\
📍 Lille, France

------------------------------------------------------------------------

## 📫 Let's Connect

```{=html}
<p align="left">
```
`<a href="https://www.linkedin.com/in/fleury-niyokwizera-2a9436291/">`{=html}
`<img src="https://img.shields.io/badge/LinkedIn-Connect%20with%20me-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>`{=html}
`</a>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

⭐ **If you find this project interesting, feel free to explore the
repository and follow my work in AI, Machine Learning and Data
Science.**
