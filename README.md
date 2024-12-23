# Gene Expression Analysis for Cancer Classification
### Author: Virendrasinh Chavda

<p align="justify">
This repository contains the code and analysis for classifying invasive and non-invasive cancer types using gene expression data. The project focuses on applying dimensionality reduction techniques and machine learning models to accurately predict cancer type, while investigating the effects of different reduction methods and validating model performance through resampling techniques. 
</p>

## Table of Contents
1. [Overview](#Overview)
2. [Data](#Data)
3. [Methodology](#Methodology)
4. [Models and Techniques](#Models-and-Techniques)
5. [Results](#Results)
6. [Contributing](#Contributing)
7. [License](#License)

## Overview
<p align="justify">
Early detection of invasive cancers is crucial in improving patient outcomes and developing effective treatment strategies. Gene expression data provides a valuable avenue for identifying patterns that differentiate between invasive and non-invasive cancers. However, gene expression datasets are often high-dimensional, making it challenging to train accurate and efficient models.

This project aims to tackle the challenge of high-dimensionality through dimensionality reduction techniques and to assess the performance of various machine learning models in classifying cancer types. The key focus is to:
</p>

1. Reduce the feature space to improve model performance and interpretability.
2. Evaluate the impact of dimensionality reduction on different supervised learning algorithms.
3. Use resampling techniques such as K-fold cross-validation and bootstrapping to ensure the robustness of the results and avoid overfitting.

<p align="justify">
By leveraging these approaches, this project demonstrates how machine learning models can achieve high accuracy in cancer classification tasks while maintaining computational efficiency and stability.
</p>

## Data:
<p align="justify">
The dataset consists of gene expression levels from 78 patients, with 4949 gene features representing expression levels. The dataset is labeled with two classes:
</p>

* Class 1: Invasive cancer
* Class 2: Non-invasive cancer

<p align="justify">
To make the analysis more computationally feasible, a random subset of 2000 genes was selected. Missing data issues were handled using k-Nearest Neighbors (kNN) imputation. The resulting dataset was then used for dimensionality reduction and machine learning classification.
</p>

## Methodology:
### 1. Preprocessing
<p align="justify">
The gene expression dataset was preprocessed to handle missing values and clean erroneous rows:
</p>

* <strong>kNN Imputation</strong>: Missing values were imputed using the kNN algorithm, which estimates missing values based on the closest data points in the feature space.
* <strong>Outlier Removal</strong>: Rows with an excessive number of missing values were removed to improve the quality of the data.
* <strong>Standardization</strong>: Gene expression levels were standardized to ensure that the models could process the data effectively.

### 2. Dimensionality Reduction
<p align="justify">
To tackle the high dimensionality of the gene expression data, several techniques were applied:
</p>

* <strong>Two-Sample t-Test</strong>: A statistical method was employed to identify genes that exhibit significant differences between the invasive and non-invasive cancer groups. This reduced the number of features from 2000 to a more manageable subset.
* <strong>LASSO Regression</strong>: L1 regularization was applied to further reduce the feature space by penalizing less important genes. This approach helps eliminate irrelevant features and improves the model's generalization.
* <strong>Variance-Based Feature Selection</strong>: Genes with high variance were selected based on the assumption that they carry the most discriminative information. This method retained genes that contribute most to class separation.

### 3. Supervised Machine Learning
<p align="justify">
Various supervised models were applied to the reduced dataset to classify the cancer types:
</p>

* <strong>Logistic Regression</strong>: A baseline model used for binary classification.
* <strong>K-Nearest Neighbors (KNN)</strong>: A distance-based classifier that showed strong performance after dimensionality reduction.
* <strong>Support Vector Machines (SVM)</strong>: A model that constructs hyperplanes to separate classes, which proved effective with high-dimensional data.
* <strong>Random Forest</strong>: An ensemble method that uses multiple decision trees to improve accuracy and robustness.
* <strong>XGBoost</strong>: A gradient-boosted decision tree algorithm optimized for high performance.

### 4. Validation and Resampling
<p align="justify">
To ensure that the models were not overfitting and could generalize to unseen data, the following resampling methods were used:
</p>

* <strong>K-fold Cross-Validation</strong>: The dataset was split into 7 folds to ensure that each model was evaluated on different subsets of the data, providing a more robust estimate of model performance.
* <strong>Bootstrapping</strong>: Multiple resamples were drawn from the original dataset to validate the stability of the model's predictions across different data distributions.

### 5. Unsupervised Learning
<p align="justify">
In addition to supervised classification, unsupervised learning techniques were applied to explore the structure of the data:
</p>

* <strong>Principal Component Analysis (PCA)</strong>: PCA was used to reduce the dimensionality further by extracting the most important principal components. This helped visualize the separation between the cancer types.
* <strong>Clustering</strong>: Hierarchical clustering and K-means clustering were performed to explore the natural groupings in the dataset and compare them with the known cancer labels.

## Models and Techniques

### Supervised Learning Models:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machines (SVM)
4. Random Forest
5. XGBoost

### Dimensionality Reduction:
1. Two-sample t-test
2. LASSO Regression
3. Variance-based Feature Selection
4. Principal Component Analysis (PCA)

### Unsupervised Learning:
1. K-Means Clustering
2. Hierarchical Clustering

## Results:
### Key Findings:

* KNN performed exceptionally well, achieving a 0 misclassification error when combined with the two-sample t-test and LASSO-reduced datasets during cross-validation. This high accuracy is attributed to the dimensionality reduction techniques that filtered out irrelevant genes.
* SVM also demonstrated strong performance, with a low misclassification rate, especially on datasets reduced by LASSO. The ability of SVM to handle high-dimensional data made it a reliable model for this task.

* <strong>Dimensionality Reduction Impact</strong>: The combination of the two-sample t-test and LASSO regression yielded the best results in terms of feature reduction without sacrificing accuracy. The feature set was reduced to just 29 key genes, allowing for efficient model training without overfitting.
* <strong>Clustering and PCA</strong>: Clustering methods, especially hierarchical clustering with complete linkage, showed a clear separation between the invasive and non-invasive classes. PCA, while useful for visualization, did not significantly improve model performance but provided insights into the underlying structure of the data.
* <strong>Resampling Stability</strong>: K-fold cross-validation and bootstrapping confirmed that the models trained on reduced data were robust and generalizable, with consistent performance across different data splits.

## Summary:
<p align="justify">
This project demonstrates the effectiveness of combining dimensionality reduction techniques with machine learning models for gene expression analysis and cancer classification. It highlights the importance of feature selection in high-dimensional datasets and provides insights into model stability through robust validation techniques.
</p>

## Contributing

Contributions are welcome! If you find any issues or want to improve the code, feel free to open a pull request or create an issue in the repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

For more details, please refer to the [project report](./Report.pdf).
