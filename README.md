# Analyzing-Diamonds-price (R)

This notebook demonstrates a machine learning workflow in R for predicting diamond prices using linear regression and regularization techniques.

linked to the notebook: https://www.kaggle.com/code/kanaksinha05/notebook644aff4695
Dataset: Diamonds dataset (/kaggle/input/diamonds/diamonds.csv)
Rows: 53,940
Columns: 10 features after removing the index column
Target variable: price (numeric) — price in USD
Predictors: carat, cut, color, clarity, depth, table, x, y, z
Source: Kaggle

1. Data Preparation
Target & Predictors Separation:

Target (y) = price

Predictors (x) = all other columns except price

Feature Scaling:

Custom minmax_scale() function to normalize numeric features to the range [0, 1]

2. Train/Test Split
Random 80/20 split using sample()

Ensures reproducibility with set.seed(42)

3. Model Training
Linear Regression (OLS): Baseline model using lm()

Lasso Regression (L1): cv.glmnet(alpha = 1) for automatic lambda selection

Ridge Regression (L2): cv.glmnet(alpha = 0) for automatic lambda selection

Additional Ridge example with a fixed penalty (lambda = 0.5)

4. Evaluation
Custom r2_score() function replicates scikit-learn’s .score() for R²

Predictions made on the test set for all models

Printed R² values for:

Without regularization (OLS)

Lasso (L1)

Ridge (L2) with lambda.min from CV

Ridge (L2) with a custom lambda value
