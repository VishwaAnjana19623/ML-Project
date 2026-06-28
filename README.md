# Obesity Level Prediction using Machine Learning

## Overview
This project work deals with the prediction of **obesity levels** through lifestyle, eating habits, and exercise using Machine Learning and Deep Learning approaches.
Models have been devised and compared in order to find out the best possible approach to classify obesity levels in multi-class classification.

## Objectives
* Determine weight classifications utilizing personal health and behavioral data  
* Implement robust data cleaning and preparation pipelines  
* Assess and benchmark several machine learning algorithms  
* Fine-tune a CatBoost Classifier to achieve maximum predictive accuracy

## Dataset Features
The dataset includes:
- Gender, Age, Height, Weight  
- Family history of overweight  
- Eating habits (FAVC, FCVC, CAEC)  
- Physical activity (FAF)  
- Water consumption (CH2O)  
- Technology usage (TUE)  
- Transportation type (MTRANS)

Target Variable:
* Insufficient Weight  
* Normal Weight  
* Overweight Level I  
* Overweight Level II  
* Obesity Type I  
* Obesity Type II  
* Obesity Type III

## Data Preprocessing
* Removed duplicate records to prevent biased learning
* Feature Engineering:
  * BMI = Weight / Height²
  * BMI × Age
  * Weight / Height
* Encoding:
 * Label Encoding for binary variables (Gender, SMOKE, etc.)
 * Ordinal Encoding for ordered categorical data (CAEC, CALC)
 * One-Hot Encoding for transportation types (MTRANS)

* Missing Values:
  * Handled using KNN Imputer (k=5) to preserve feature relationships.
* Feature Scaling:
  * StandardScaler
* Train-Test Split:
  * 80% Training / 20% Testing. Applied strict stratify=y parameters to ensure perfectly balanced class representation

## Models Used
- Random Forest  
- Support Vector Machine (SVM)  
- CatBoost Classifier *(My Contribution)* 
- Deep Neural Network (DNN)

## CatBoost Classifier
* A powerful gradient-boosting algorithm naturally optimized for datasets with complex categorical variables.
* Loss Function:
  * Configured to MultiClass, which is critical for measuring errors accurately across the 7 different target categories.
* Clean Execution:
  * verbose=0 applied to mute console output during the intensive training loops.

## Model Optimization
* Tuning Method: 
 * Automated via RandomizedSearchCV to systematically test hyperparameter combinations.  
* Validation: 3-Fold Cross-Validation (cv=3) utilized to ensure reliable, unbiased scoring. 
* Optimal Hyperparameters Discovered:
 * Depth:6   
 * Iterations: 505  
 * Learning Rate: ~0.061  
 * L2 Leaf Reg: 1

## Results
- Final Test Accuracy: ~95.7%
- Strong performance across all obesity classes  
- Balanced precision, recall, and F1-score

## Evaluation Metrics
- Family history and diet habits strongly influence obesity  
- Physical activity plays a major role  
- Transportation type also contributes to prediction

## Technologies Used
* Python   
* Pandas, NumPy 
* Scikit-learn  
* CatBoost 
* TensorFlow
Matplotlib, Seaborn  


