# 🌦️ Weather Type Prediction 

## 📌 Project Overview

This repository contains a complete end-to-end workflow for predicting weather type (e.g., sunny,  cloudy, rainy and snowy) based on meteorological features such as temperature, wind speed, humidity, precipitation, pressure, and visibility. The project includes:

- Exploratory Data Analysis (EDA): Data cleaning, visualization, and feature understanding.

- Feature Engineering: Creation of new numerical and categorical features, handling of skewed distributions, and scaling/encoding for machine learning models.

- Modeling: Baseline and final models including Logistic Regression, Random Forest, and Gradient Boosting, with pipelines for preprocessing and training.

- Cross-Validation: Stratified cross-validation to assess model stability and performance consistency.

- Evaluation: Metrics (precision, recall, F1-score, ROC-AUC) and confusion matrices for both training and unseen test data.

- Reproducibility: Preprocessing pipelines, label encoding, and metrics saved for consistent evaluation.

The final Logistic Regression model was selected for deployment based on stable performance, interpretability, and generalization capability.

## 🗂️ Project Structure

This project follows a reproducible structure:
- **EDA Notebook:** Exploratory Data Analysis and initial insights
- **Baseline Modeling Notebook:** Training and evaluation of initial models
- **Retraining Notebook:** Retrain and evaluate the models after feature engineering
- **Cross-Validation and Final Evaluation Notebook:** Model stability assessment using Stratified CV. Final training and unseen test evaluation
- **Scripts:**
	- **train.py:** Model training logic
	- **evaluate.py:** Evaluation and metrics computation in both sets to rule out overfitting
	- **feature_eng.py:** More descriptive features were created based on the existing ones
	- **preprocessing.py:** Preprocessors are built here, using a StandardScaler for numerical features, and One Hot encoding for categorical ones.
	- **encoding.py:** Global label target encoder
	- **validation.py:** Stratified cross validation methodology
	- **utils.py:** Some useful functions for loading the data, build some charts and compute mutual information.
- **Results:** Stored metrics, plots, and confusion matrices per step in the project
- **Data:** Raw, cleaned and featured data
## 🔍 Exploratory Data Analysis (EDA)

Key findings:

- The target variable is **multiclass**, requiring careful handling of class distribution.
    
- Several numerical features showed **right-skewed distributions**.
    
- Some features demonstrated strong correlation with the target.
    
- Categorical variables had **low cardinality**, making them suitable for One Hot encoding.

These insights guided model selection and feature engineering decisions.

## 🤖 Models Trained

Three supervised classification models were evaluated:

1. **Logistic Regression** (Final Selected Model)
    
2. Random Forest 
    
3. Gradient Boosting
    

Preprocessing steps included:

- Scaling numerical features (for Logistic Regression)
    
- One-Hot Encoding categorical features
    
- Label Encoding of the target variable
## 🛠️ Feature Engineering

Several improvements were applied to enhance model performance:

- Creation of **new numerical features** derived from existing variables
    
- **Rescaling of skewed features** to improve linear model performance
    
- Creation of **new categorical features** from numerical bins

After feature engineering, all models showed improved performance, with **Logistic Regression remaining the best-performing model**.

## 🔁 Cross Validation

Cross-validation was conducted to ensure model reliability and stability:

- **Stratified K-Fold Cross Validation** was used to preserve class distribution
    
- CV was performed only on the **training set**, keeping the test set unseen
    
- Metrics across folds were visualized using a **boxplot** to assess variance
    

Metrics evaluated:

- Precision
    
- Recall
    
- F1-score
    
- ROC-AUC
## 📊 Final Evaluation

The final Logistic Regression model was:

- Retrained on the full training dataset
    
- Evaluated on the untouched test set
    
- Assessed using confusion matrices and performance metrics
    
- Results saved for reproducibility and reporting
    

This confirms the model's generalization ability and ensures an unbiased final performance estimate.
## 🏁 Conclusion

- Logistic Regression outperformed tree-based models consistently
    
- Feature engineering significantly improved performance
    
- Cross-validation confirmed model stability
    
- No signs of overfitting were observed
    
- The final model demonstrates strong predictive performance and reliability


This project provides a complete, end-to-end machine learning workflow suitable for academic or professional applications.

## 🧰 Technologies Used

- Python
    
- pandas, numpy
    
- scikit-learn
    
- matplotlib, seaborn
    
- Jupyter Notebook
## 📁 Results

All evaluation outputs are stored in the `results/` directory, including:

- Metrics tables (CSV)
    
- Confusion matrices
    
- Cross-validation summaries
    
- Cross-validation stability plot



