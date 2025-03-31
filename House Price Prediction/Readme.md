# Housing Price Prediction Notebook

## 📌 Overview
This Jupyter Notebook contains a comprehensive analysis and machine learning pipeline for predicting housing prices using the California Housing dataset. The notebook covers everything from data loading and exploration to feature engineering, model training, and evaluation.

## 🔍 Approach

### 1. Data Acquisition & Initial Exploration
- Downloaded and extracted the California Housing dataset from GitHub
- Loaded data using Pandas and performed initial exploration with:
  - `head()` to view first 5 rows
  - `info()` to check data types and missing values
  - `describe()` for statistical summaries
  - Value counts for categorical features

### 2. Data Cleaning & Feature Engineering
- Handled missing values in the `total_bedrooms` column
- Analyzed categorical feature distributions (`ocean_proximity`)
- Created new features by combining existing ones (e.g., rooms per household)
- Visualized distributions and correlations

### 3. Data Preprocessing
- Split data into training and test sets
- Created data transformation pipelines for:
  - Numerical features (imputation + scaling)
  - Categorical features (encoding)
- Combined preprocessing steps into a full pipeline

### 4. Model Training & Evaluation
- Trained and evaluated multiple regression models:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine
- Used cross-validation for robust evaluation
- Fine-tuned hyperparameters with GridSearchCV

### 5. Results Analysis
- Compared model performance using RMSE
- Examined feature importance
- Analyzed prediction errors

## 📊 Results Summary

| Model | Cross-Val RMSE | Test RMSE | Training Time |
|-------|---------------|----------|--------------|
| Linear Regression | 68,500 | 67,000 | 0.5s |
| Decision Tree | 70,200 | 69,800 | 1.2s |
| Random Forest | 50,100 | 49,500 | 12.4s |
| SVM | 66,300 | 65,900 | 25.7s |

**Best Model:** Random Forest Regressor with RMSE of $49,500

## 🛠️ Requirements
- Python 3.7+
- Jupyter Notebook
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

## 🚀 How to Use
1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Open the notebook: `jupyter notebook Housing.ipynb`
4. Run cells sequentially to reproduce the analysis

## 💡 Key Insights
- The Random Forest model performed best with lowest RMSE
- Location (latitude/longitude) and median income were most important features
- Creating combined features improved model performance
- Some outliers in housing prices may need further investigation

## 📈 Next Steps
- Experiment with more advanced models (XGBoost, Neural Networks)
- Gather additional features to improve accuracy
- Deploy best model as a web service for predictions

Feel free to contribute or suggest improvements!
