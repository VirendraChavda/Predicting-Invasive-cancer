# Gene Expression Analysis for Cancer Classification
### Author: Virendrasinh Chavd

** This repository contains the code and analysis for classifying invasive and non-invasive cancer types using gene expression data. The project focuses on applying dimensionality reduction techniques and machine learning models to accurately predict cancer type, while investigating the effects of different reduction methods and validating model performance through resampling techniques. **

## Table of Contents
* Overview
* Data
* Methodology
* Models and Techniques
* Results
* Installation
* Usage

## Overview
Early detection of invasive cancers is crucial in improving patient outcomes and developing effective treatment strategies. Gene expression data provides a valuable avenue for identifying patterns that differentiate between invasive and non-invasive cancers. However, gene expression datasets are often high-dimensional, making it challenging to train accurate and efficient models.

This project aims to tackle the challenge of high-dimensionality through dimensionality reduction techniques and to assess the performance of various machine learning models in classifying cancer types. The key focus is to:

Reduce the feature space to improve model performance and interpretability.
Evaluate the impact of dimensionality reduction on different supervised learning algorithms.
Use resampling techniques such as K-fold cross-validation and bootstrapping to ensure the robustness of the results and avoid overfitting.
By leveraging these approaches, this project demonstrates how machine learning models can achieve high accuracy in cancer classification tasks while maintaining computational efficiency and stability.

Data
The dataset consists of gene expression levels from 78 patients, with 4949 gene features representing expression levels. The dataset is labeled with two classes:

Class 1: Invasive cancer
Class 2: Non-invasive cancer
To make the analysis more computationally feasible, a random subset of 2000 genes was selected. Missing data issues were handled using k-Nearest Neighbors (kNN) imputation. The resulting dataset was then used for dimensionality reduction and machine learning classification.

Methodology
1. Preprocessing
The gene expression dataset was preprocessed to handle missing values and clean erroneous rows:

kNN Imputation: Missing values were imputed using the kNN algorithm, which estimates missing values based on the closest data points in the feature space.
Outlier Removal: Rows with an excessive number of missing values were removed to improve the quality of the data.
Standardization: Gene expression levels were standardized to ensure that the models could process the data effectively.
2. Dimensionality Reduction
To tackle the high dimensionality of the gene expression data, several techniques were applied:

Two-Sample t-Test: A statistical method was employed to identify genes that exhibit significant differences between the invasive and non-invasive cancer groups. This reduced the number of features from 2000 to a more manageable subset.
LASSO Regression: L1 regularization was applied to further reduce the feature space by penalizing less important genes. This approach helps eliminate irrelevant features and improves the model's generalization.
Variance-Based Feature Selection: Genes with high variance were selected based on the assumption that they carry the most discriminative information. This method retained genes that contribute most to class separation.
3. Supervised Machine Learning
Various supervised models were applied to the reduced dataset to classify the cancer types:

Logistic Regression: A baseline model used for binary classification.
K-Nearest Neighbors (KNN): A distance-based classifier that showed strong performance after dimensionality reduction.
Support Vector Machines (SVM): A model that constructs hyperplanes to separate classes, which proved effective with high-dimensional data.
Random Forest: An ensemble method that uses multiple decision trees to improve accuracy and robustness.
XGBoost: A gradient-boosted decision tree algorithm optimized for high performance.
4. Validation and Resampling
To ensure that the models were not overfitting and could generalize to unseen data, the following resampling methods were used:

K-fold Cross-Validation: The dataset was split into 7 folds to ensure that each model was evaluated on different subsets of the data, providing a more robust estimate of model performance.
Bootstrapping: Multiple resamples were drawn from the original dataset to validate the stability of the model's predictions across different data distributions.
5. Unsupervised Learning
In addition to supervised classification, unsupervised learning techniques were applied to explore the structure of the data:

Principal Component Analysis (PCA): PCA was used to reduce the dimensionality further by extracting the most important principal components. This helped visualize the separation between the cancer types.
Clustering: Hierarchical clustering and K-means clustering were performed to explore the natural groupings in the dataset and compare them with the known cancer labels.
Models and Techniques
Supervised Learning Models:
Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machines (SVM)
Random Forest
XGBoost
Dimensionality Reduction:
Two-sample t-test
LASSO Regression
Variance-based Feature Selection
Principal Component Analysis (PCA)
Unsupervised Learning:
K-Means Clustering
Hierarchical Clustering
Results
Key Findings:
KNN performed exceptionally well, achieving a 0 misclassification error when combined with the two-sample t-test and LASSO-reduced datasets during cross-validation. This high accuracy is attributed to the dimensionality reduction techniques that filtered out irrelevant genes.
SVM also demonstrated strong performance, with a low misclassification rate, especially on datasets reduced by LASSO. The ability of SVM to handle high-dimensional data made it a reliable model for this task.
Dimensionality Reduction Impact: The combination of the two-sample t-test and LASSO regression yielded the best results in terms of feature reduction without sacrificing accuracy. The feature set was reduced to just 29 key genes, allowing for efficient model training without overfitting.
Clustering and PCA: Clustering methods, especially hierarchical clustering with complete linkage, showed a clear separation between the invasive and non-invasive classes. PCA, while useful for visualization, did not significantly improve model performance but provided insights into the underlying structure of the data.
Resampling Stability: K-fold cross-validation and bootstrapping confirmed that the models trained on reduced data were robust and generalizable, with consistent performance across different data splits.

Summary:
This project demonstrates the effectiveness of combining dimensionality reduction techniques with machine learning models for gene expression analysis and cancer classification. It highlights the importance of feature selection in high-dimensional datasets and provides insights into model stability through robust validation techniques.
