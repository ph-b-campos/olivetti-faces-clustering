# 👤 Olivetti Faces: Clustering & Classification

**Executive Summary:**
In this computer vision project, I built a Machine Learning pipeline to identify individuals in the Olivetti Faces dataset. Beyond standard multiclass classification, this project heavily explores unsupervised learning techniques—specifically Principal Component Analysis (PCA) and $k$-means clustering—as dimensionality reduction and feature engineering tools to evaluate their impact on classifier performance.

---

## 1. The Technical Challenge

Facial recognition is a high-dimensional problem. The challenge in this specific dataset is the "curse of dimensionality" combined with a very small sample size. 



**Objective:** Predict which of the 40 distinct subjects is represented in a given picture.
* **Dataset:** 400 images (10 different images for each of the 40 subjects, varying lighting and facial expressions).
* **Dimensionality:** Each image is 64x64 pixels, flattened into an array of 4,096 features.
* **Problem Type:** Supervised Learning (Multiclass Classification)
* **Target Metric:** Accuracy

## 2. Methodology & Feature Engineering

To tackle the high dimensionality and evaluate different feature representations, I implemented the following analytical pipeline:

* **Data Splitting:** Applied stratified sampling to ensure the training and test sets had an equal representation of all 40 subjects.
* **Noise Reduction (PCA):** Applied PCA to preserve 99% of the variance, compressing the feature space and removing background noise before clustering.
* **Clustering Analysis:** Used $k$-means to group similar faces. I utilized the Elbow Method (Inertia) and Silhouette Scores to algorithmically determine the optimal number of clusters ($k=101$).

* **Pipeline Integration & Feature Union:** Built robust Scikit-Learn pipelines to test three approaches:
  1. Training on raw pixel data (Baseline).
  2. Training exclusively on the reduced dimensional space (PCA + K-Means cluster distances).
  3. Using `FeatureUnion` to append the cluster distances to the original raw pixels.

## 3. Machine Learning Models & Tech Stack

The classification phase focused on linear models and distance-based algorithms, utilizing `RandomizedSearchCV` to fine-tune both the classifier hyperparameters (like `C` for Logistic Regression) and the preprocessing steps (like `n_clusters` for K-Means) simultaneously within the pipeline.

* **Baseline:** Logistic Regression (One-Vs-Rest).
* **Alternative Algorithms:** K-Nearest Neighbors (KNN).

🛠️ **Tech Stack:**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

> 💡 **Did the complex clustering pipeline beat the baseline model? Read the detailed Error Analysis, visual cluster representations, and final conclusions directly inside the Jupyter Notebook.**
