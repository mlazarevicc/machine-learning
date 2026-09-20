# Machine Learning Portfolio

Welcome to my Machine Learning portfolio! This repository contains a collection of end-to-end data science and machine learning projects. These projects demonstrate my ability to handle various data types (tabular, textual) and apply different modeling techniques (Supervised and Unsupervised Learning) to solve complex problems.

## Repository Structure

This repository is divided into three main projects, each housed in its own directory with dedicated data, source code, and detailed reports.

### 1. [Real Estate Price Prediction (Regression)](./real_estate_regression)

* **Goal:** Predict real estate prices based on property features and textual descriptions.

* **Key Techniques:** Robust preprocessing, Target Encoding for high-cardinality categorical variables, feature engineering (handling textual indicators like 'lux' or 'renovated'), and `Ridge` regression.

* **Highlights:** Implementation of a robust `scikit-learn` Pipeline with a `TransformedTargetRegressor` to handle log-normal price distributions.

### 2. [Boardgame Post Classification (NLP / Classification)](./boardgames_classification)

* **Goal:** Classify Serbian forum posts into one of five board game categories.

* **Key Techniques:** Natural Language Processing (NLP), Character N-Grams TF-IDF vectorization, handling class imbalance (`class_weight='balanced'`), and Ensemble Learning (Soft Voting).

* **Highlights:** Built a custom multi-model ensemble combining linguistic logic, raw forum slang, and temporal metadata to achieve high Macro F1 scores even on rare classes.

### 3. [Pokemon Dataset Analysis (Clustering)](./pokemon_clustering)

* **Goal:** Group 588 Pokémon entities based on their combat statistics and experience parameters using unsupervised learning.

* **Key Techniques:** Dimensionality reduction, exhaustive hyperparameter tuning (GridSearch), K-Means, Gaussian Mixture Models (GMM), and DBSCAN.

* **Highlights:** Comprehensive exploratory data analysis to handle multimodal distributions and internal validation using the Silhouette score.

## Tech Stack & Tools

* **Language:** Python 3

* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`

* **Concepts:** Supervised & Unsupervised Learning, Feature Engineering, Regularization, Text Vectorization (TF-IDF), Pipeline Automation.

## How to Run

Each sub-directory contains a `README.md` with detailed project reports, an executable `.py` script for training/predicting, and a Jupyter Notebook (`.ipynb`) for exploratory data analysis (EDA).

To run a specific project script (e.g., Real Estate Regression), navigate to its directory and run:

```
python train.json test.json
```

*Feel free to explore the individual folders for deep dives into the EDA and mathematical justifications for the chosen models!*
