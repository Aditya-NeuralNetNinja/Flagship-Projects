## 💻 Laptop Price Estimation: A Regression Approach

## 💼 Overview
This project focuses on predicting laptop prices using a regression-based approach. The workflow encompasses exploratory data analysis, feature engineering, and extensive data preprocessing to handle categorical variables, extract key features, and transform memory-related columns. Various regression models including Linear Regression, Ridge and Lasso Regression, KNN, Decision Tree, SVM, Random Forest, ExtraTrees, AdaBoost, Gradient Boost, XGBoost, LightGBM, CatBoost, Voting Regressor, and Stacking Regressor were employed and evaluated based on R-squared score and Mean Absolute Error. The Stacking Regressor emerged as the top-performing model, yielding the most accurate predictions. The finalized model is exported for future use as df.pkl (containing the dataset) and pipe.pkl (housing the trained Stacking Regressor). This comprehensive project on GitHub presents an in-depth analysis of the dataset, robust model development, and a detailed evaluation process, providing a practical solution for estimating laptop prices based on their specifications.

## 📋 Description
The project involves the following steps:

### 📦 Imports
Python libraries such as NumPy, Pandas, Matplotlib, Seaborn, and warnings are imported.

### 📊 Load Data
The dataset is loaded using Pandas from the specified file path.

### 🔍 Exploratory Data Analysis (EDA)
Various exploratory data analysis steps are performed:
- Displaying data samples
- Checking data shape and information
- Dropping unnecessary columns
- Feature engineering (modifying columns like Ram, Weight, ScreenResolution, etc.)
- Displaying plots and correlation of features

### 🛠️ Feature Engineering
- Typecasting columns to appropriate data types
- Feature extraction from existing columns
- Handling and transforming memory-related columns (SSD, HDD, etc.)
- Encoding categorical variables

### 🏗️ Model Building
Utilizing various regression models:
- Linear Regression
- Ridge and Lasso Regression
- KNN, Decision Tree, SVM, Random Forest, ExtraTrees, AdaBoost, Gradient Boost, XGBoost, LightGBM, CatBoost
- Voting Regressor and Stacking Regressor

## 📊 Model Evaluation
Each model's performance is evaluated using metrics like R-squared score and Mean Absolute Error (MAE).

### 🥇 Best Model
The Stacking Regressor model exhibited the highest R-squared score among all the models.

## 📤 Exporting Model
The Stacking Regressor model is exported using pickle for future use.

## 🧪 Testing the Model
An example input is tested with the model to predict a laptop's price based on its specifications.

## 📁 Files
- `df.pkl`: Pickled file containing the dataset
- `pipe.pkl`: Pickled file containing the trained Stacking Regressor model

Feel free to explore the code and modify it according to your requirements!
