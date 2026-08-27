# House Prices: Advanced Regression Techniques

This project builds an advanced machine learning pipeline to solve the classic Kaggle housing data challenge. The goal is to predict the final sale price (SalePrice) of residential homes.


## Project Overview

Predicting house prices is a complex challenge because real-world data contains a mix of continuous numeric values (like square footage), ranked text metrics (like kitchen quality), and unranked text features (like neighborhood names). Additionally, many entries contain missing data (`NaN`).

To achieve high accuracy and utilize the full breadth of the data, this project moves past basic linear models and employs an end-to-end processing pipeline paired with an advanced **Gradient Boosting Regressor** algorithm.


## Step-by-Step Workflow Explanation

### 1. Data Loading & Inspection
* **Action:** Loaded the dataset using the `pandas` library (pd.read_csv).
* **Why:** We used structural inspections (`df.info()`) to reveal the shape of the dataset and catalog which columns contained numbers versus text strings.

### 2. Feature & Target Splitting
* **Action:** Extracted the target variable `y` (SalePrice) and created our feature matrix `X` by dropping SalePrice and the tracking Id column.
* **Why:** The `Id` column is a sequential counter with no predictive power. Leaving it in causes **overfitting**, where a machine learning model accidentally builds fake rule patterns based on arbitrary spreadsheet row positions instead of actual housing characteristics.

### 3. Target Distribution Visualization
* **Action:** Plotted a histogram of the target variable SalePrice using matplotlib.
* **Why:** Visualizing the target allows us to understand the skewness and range of house prices across the dataset, ensuring a clear conceptual baseline of the values we want to forecast.

### 4. Categorical Encoding Exploration
Before choosing a final approach, we explored multiple industry-standard techniques to convert text features into numbers:
* **Label Encoding:** Loops through columns and replaces distinct text values with standalone numbers (`0, 1, 2...`). This is ideal for simple categorical transformations.
* **One-Hot Encoding:** Creates completely unique binary (`0` or `1`) columns for every individual category option. This prevents models from assuming non-existent numerical relationships between distinct categories (e.g., thinking a neighborhood label of `3` is inherently three times larger than a neighborhood label of `1`).

### 5. Automated Preprocessing Pipeline
To prevent data leakage and handle dirty data seamlessly, we structured a unified ColumnTransformer:
* **Numeric Columns:** Missing values are automatically filled (SimpleImputer) with the column's **median** to prevent outliers from distorting the dataset.
* **Categorical Columns:** Missing values are filled with the **most frequent** text occurrence (the mode), followed by an automated OneHotEncoder to prepare the data fields cleanly for math execution.

### 6. The Advanced Regression Model: Gradient Boosting
Instead of simple linear regression, we implemented a **Gradient Boosting Regressor** as our predictive engine. 
* **How it Works:** Gradient Boosting is an ensemble learning method. It begins by creating a basic decision tree to predict house values. It calculates the errors of that tree, then sequentially creates a *second* tree specifically tailored to correct the mistakes of the first. It repeats this pattern hundreds of times, building a highly optimized chain of trees where each model learns directly from the shortcomings of its predecessor.

### 7. Evaluation & Validation Setup
* **Action:** Split our data into an 80% training set and a 20% validation set using Scikit-Learn's train_test_split.
* **Evaluation Metric:** Used **Mean Absolute Error (MAE)** to score the model's accuracy. MAE tells us exactly how many dollars our house predictions are off by on average when compared against real, known transaction numbers.


