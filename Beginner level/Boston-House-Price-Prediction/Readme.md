# 🏡 Boston House Price Prediction

### Machine Learning Engineer Internship — ShadowFox

A supervised machine learning regression project that predicts Boston house prices using housing and socioeconomic features from the Boston Housing dataset.

---

## 📌 1. Project Overview

The objective of this project is to build and evaluate machine learning regression models that can predict house prices based on different housing-related features.

Five regression algorithms were implemented and compared to identify the best-performing model.

## 🎯 2. Project Objectives

- Understand and explore the dataset.
- Perform data cleaning and preprocessing.
- Analyze relationships between features and house prices.
- Train multiple regression models.
- Evaluate model performance using appropriate regression metrics.
- Compare models and select the best overall performer.

## 📂 3. Dataset Information

The project uses the Boston Housing dataset.

| Property | Details |
|---|---|
| Total records | 506 |
| Total columns | 14 |
| Input features | 13 |
| Target variable | MEDV |
| Problem type | Supervised Regression |

### Features

- CRIM — Per capita crime rate
- ZN — Residential land zoned for large lots
- INDUS — Non-retail business acres
- CHAS — Charles River dummy variable
- NOX — Nitric oxide concentration
- RM — Average number of rooms
- AGE — Proportion of older owner-occupied units
- DIS — Weighted distance to employment centres
- RAD — Accessibility to radial highways
- TAX — Property tax rate
- PTRATIO — Pupil-teacher ratio
- B — Dataset's demographic-derived feature
- LSTAT — Lower-status population percentage

**Target variable:** `MEDV` — Median value of owner-occupied homes.

## 🧹 4. Data Preprocessing

The following preprocessing and data-cleaning activities were performed:

- Checked dataset dimensions and data types.
- Checked and handled missing values.
- Checked duplicate records.
- Investigated potential outliers using the IQR method.
- Analyzed the distribution of the target variable.
- Separated independent variables (`X`) and target variable (`y`).
- Split the dataset into training and testing sets.

### Train-Test Split

- Training data: 80% (404 records)
- Testing data: 20% (102 records)
- Random state: 42

## 📊 5. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify important patterns.

Key observations:

- RM showed a positive relationship with MEDV.
- LSTAT showed a strong negative relationship with MEDV.
- PTRATIO also showed a negative relationship with MEDV.
- Correlation analysis helped identify important relationships between features.

Visualizations included:
- Target variable distribution.
- Feature distributions.
- Correlation heatmap.
- Scatter plots.
- Outlier analysis.

## 🤖 6. Machine Learning Models

Five regression algorithms were implemented:

| Model | Description |
|---|---|
| Linear Regression | Fits a linear relationship between features and target |
| Decision Tree Regressor | Uses decision rules to predict continuous values |
| Gradient Boosting Regressor | Combines sequentially trained weak learners |
| Random Forest Regressor | Combines predictions from multiple decision trees |
| Support Vector Regression (SVR) | Uses support vector methods to predict continuous values |

## 📏 7. Evaluation Metrics

The models were evaluated using three regression metrics.

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted values. Lower is better.

**Mean Squared Error (MSE)**

Measures the average squared prediction error. Lower is better, with larger errors penalized more heavily.

**R² Score**

Measures how much of the variation in the target variable is explained by the model. Higher is better.

## 🏆 8. Model Comparison Results

The following results were obtained from the test dataset:

| Model | MAE | MSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 3.1487 | 24.9994 | 0.6591 |
| Decision Tree | 2.7776 | 12.4201 | 0.8306 |
| Gradient Boosting | 2.3624 | 12.4420 | 0.8303 |
| **Random Forest** | **2.2140** | **9.1504** | **0.8752** |
| SVR | 2.0810 | 13.9944 | 0.8092 |

### Best Overall Model: Random Forest Regressor

Random Forest achieved:

- MAE: 2.2140
- MSE: 9.1504
- R² Score: 0.8752

It achieved the lowest MSE and highest R² score among the five models.

SVR achieved the lowest MAE of 2.0810.

Based on the combined evaluation, Random Forest was selected as the best overall model.

## 📈 9. Visualizations

The project includes visualizations to understand the dataset and compare model performance.

Model comparison charts:
- MAE Comparison
- MSE Comparison
- R² Score Comparison

Additional EDA visualizations are available in the project notebook.

## 🛠️ 10. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

## ▶️ 11. How to Run the Project

**Step 1:** Clone the repository.

```bash
git clone <your-github-repository-url>
```

**Step 2:** Open the project folder.

```bash
cd <your-project-folder>
```

**Step 3:** Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**Step 4:** Launch Jupyter Notebook.

```bash
jupyter notebook
```

**Step 5:** Open the Boston House Price Prediction notebook and run the cells in order.

Make sure the dataset is available at the file path used in the notebook.

## 🔍 12. Limitations and Future Improvements

Possible future improvements include:

- Hyperparameter tuning.
- Cross-validation.
- More systematic feature selection.
- Improved preprocessing using pipelines fitted only on training data.
- Testing on additional datasets.
- Further model optimization.

**Evaluation note:** For a rigorous final evaluation, missing-value imputation must be fitted using training data only and then applied to the test data. If imputation was performed on the complete dataset before splitting, the reported test results may be somewhat optimistic.

## 🎓 13. Learning Outcomes

Through this project, I gained practical experience in:

- Understanding supervised machine learning.
- Cleaning and exploring real-world datasets.
- Preparing data for regression problems.
- Training multiple machine learning models.
- Evaluating and comparing regression performance.
- Identifying a suitable model for a prediction task.
- Organizing a machine learning project for GitHub.

## 👨‍💻 14. Author

**Name:** [Your Name]  
**Internship:** Machine Learning Engineer Intern  
**Organization:** ShadowFox

---

⭐ If you find this project useful, feel free to explore the repository.

**Thank you for visiting!**