# Car Selling Price Prediction

## 📌 Project Overview

This project focuses on predicting the selling price of used cars using Machine Learning regression techniques. The objective is to understand the factors influencing car prices, perform exploratory data analysis, prepare the dataset, train multiple regression models, and identify the best-performing model.

This project was completed as part of the **ShadowFox Machine Learning Engineer Internship – Intermediate Level**.

## 🎯 Objectives

- Analyze the used car dataset and understand its features.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Perform feature engineering and feature selection.
- Train and compare multiple regression models.
- Apply hyperparameter tuning using RandomizedSearchCV.
- Evaluate model performance using regression metrics.
- Identify important features influencing car selling prices.

## 📂 Project Structure

```text
Car-Selling-Price-Prediction/
│
├── data/
│   └── car_data.csv
│
├── notebooks/
│   └── Car_Selling_Price_Prediction.ipynb
│
├── outputs/
│   └── figures/
│
├── best_car_price_model.pkl
│
└── README.md
```

## 📊 Dataset Description

The dataset contains information about used cars, including:

| Feature | Description |
|---|---|
| Car_Name | Name of the car |
| Year | Year of manufacture |
| Selling_Price | Selling price of the car (target variable) |
| Present_Price | Current ex-showroom price |
| Kms_Driven | Distance driven by the car |
| Fuel_Type | Type of fuel used |
| Seller_Type | Type of seller |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |

- Original dataset: 301 records and 9 columns.
- Duplicate records removed: 2.
- Final dataset: 299 records.
- Missing values: None detected.

## 🧹 Data Preprocessing and Feature Engineering

The following preprocessing steps were performed:

- Inspected dataset structure, data types, and statistical information.
- Checked for missing values and duplicate records.
- Removed two duplicate records.
- Created a new feature, `Car_Age`, using the reference year 2018.
- Excluded `Car_Name` from model inputs due to its high cardinality.
- Excluded the original `Year` feature after deriving `Car_Age`.
- Applied One-Hot Encoding to categorical features.
- Split the dataset into 80% training data and 20% testing data.

### Selected Features

- Present_Price
- Kms_Driven
- Owner
- Car_Age
- Fuel_Type
- Seller_Type
- Transmission

**Target Variable:** Selling_Price

## 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand numerical and categorical distributions and relationships between features.

Visualizations included:

- Histograms and count plots.
- Scatter plots.
- Box plots.
- Correlation heatmap.
- Pair plot.

The analysis showed a strong positive relationship between Present_Price and Selling_Price. Car age also showed an influence on selling prices.

## 🤖 Machine Learning Models

Five regression algorithms were trained and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regressor (SVR)

### Model Evaluation Metrics

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

## ⚙️ Hyperparameter Tuning

RandomizedSearchCV with 5-fold cross-validation was used to tune:

- Random Forest Regressor
- Gradient Boosting Regressor

Different hyperparameter combinations were evaluated to identify suitable model configurations.

Hyperparameter tuning improved the Random Forest test results compared with its baseline, but the tuned models did not outperform the baseline Gradient Boosting model on the test dataset.

## 📊 Model Comparison

The following results were obtained:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Gradient Boosting (Baseline) | 1.1349 | 2.5232 | **0.7530** |
| Linear Regression | 1.4729 | 2.5240 | 0.7528 |
| Decision Tree | 1.3273 | 2.9130 | 0.6708 |
| Gradient Boosting (Tuned) | 1.3016 | 3.0715 | 0.6339 |
| Random Forest (Tuned) | 1.3851 | 3.2659 | 0.5862 |
| Random Forest (Baseline) | 1.5270 | 3.7082 | 0.4665 |
| SVR (Baseline) | 3.1295 | 5.3217 | -0.0988 |

## 🏆 Final Model Selection

The **Baseline Gradient Boosting Regressor** was selected as the best-performing model based on the observed test-set results.

Performance:

- MAE: 1.1349
- MSE: 6.3667
- RMSE: 2.5232
- R² Score: 0.7530

It achieved the highest R² score and lowest RMSE among the evaluated models.

## 🔍 Feature Importance Analysis

Feature importance analysis was performed using the Gradient Boosting model.

Key findings:

- Present_Price: approximately 89.85%
- Car_Age: approximately 7.49%
- Kms_Driven: approximately 0.97%

Present_Price was the most influential feature, followed by Car_Age.

## ⚠️ Limitations

- The dataset contains a relatively small number of records.
- The baseline SVR model was trained without feature scaling, which may have affected its performance.
- Initial model comparisons used the test-set results, so these results should be treated as exploratory rather than an unbiased final performance estimate.
- The dataset may not represent all used-car markets or current market conditions.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## ▶️ How to Run the Project

1. Clone this repository.
2. Open the project folder.
3. Install the required Python libraries.
4. Place the dataset inside the `data/` directory.
5. Open `Car_Selling_Price_Prediction.ipynb` in Jupyter Notebook or VS Code.
6. Run the notebook cells in sequence.

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib
```

## 👨‍💻 Internship

**Program:** ShadowFox Machine Learning Engineer Internship

**Level:** Intermediate

**Project:** Car Selling Price Prediction