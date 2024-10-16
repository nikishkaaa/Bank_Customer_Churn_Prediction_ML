# Project Overview: Bank Customer Churn Prediction

**Problem Statement:**
Customer churn is a critical issue for banks, as it directly impacts revenue and growth. This project aims to predict whether a customer will leave the bank using a dataset containing various customer attributes and transaction history. By accurately predicting churn, the bank can implement targeted retention strategies, thereby improving customer satisfaction and reducing loss.


### Project Description

This project implements a customer churn prediction system for a banking institution using various machine learning models for binary classification. The primary goal is to identify customers who are likely to leave the bank, enabling proactive retention strategies. 

The dataset consists of multiple customer features, including demographic information and account details, which are analyzed to extract valuable insights. After data preprocessing, a variety of models are trained and evaluated to find the optimal solution. The project highlights the importance of model selection and tuning in achieving the best predictive performance.

By leveraging machine learning, this project provides a framework for understanding customer behavior, ultimately assisting banks in making informed decisions to enhance customer loyalty and retention.


**Approach:**
In this project, we applied various machine learning models for binary classification to determine the likelihood of customer churn. The approach includes the following steps:

1. **Data Preprocessing:** 
   - Loading and exploring the dataset to understand its structure.
   - Handling missing values and categorical features through encoding and imputation.
   - Normalizing and scaling numerical features for better model performance.

2. **Model Selection:**
   - Experimenting with different machine learning algorithms including Logistic Regression, Decision Trees, XGBoostClassifier, Support Vector Machines (SVM), KNN and LightGBM.
   - Splitting the data into training and testing sets to ensure unbiased evaluation of model performance.

3. **Model Evaluation:**
   - ROC-AUC: The area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between churn and non-churn classes

5. **Hyperparameter Tuning:**
   - Employing techniques like Grid Searchto optimize model parameters for improved accuracy.

6. **Final Model Selection:**
   - Comparing the results from different models to select the best-performing algorithm based on evaluation metrics.
  

### Results

<table>
  <tr>
    <th colspan="2">Logistic Regression</th>
    <th colspan="2">Decision Tree</th>
    <th colspan="2">XGBoostClassifier</th>
    <th colspan="2">LightGBM</th>
    <th colspan="2">KNN</th>
    <th colspan="2">SVM</th>
  </tr>
  <tr>
    <th>Train ROC AUC score</th>
    <th>Test ROC AUC score</th>
    <th>Train ROC AUC score</th>
    <th>Test ROC AUC score</th>
    <th>Train ROC AUC score</th>
    <th>Test ROC AUC score</th>
    <th>Train ROC AUC score</th>
    <th>Test ROC AUC score</th>
    <th>Train ROC AUC score</th>
    <th>Test ROC AUC score</th>
    <th>Train ROC AUC score</th>
    <th>Test ROC AUC score</th>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>


