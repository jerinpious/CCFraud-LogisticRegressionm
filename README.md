# Credit Card Fraud Detection Using Logistic Regression

## Project Description
This project aims to detect fraudulent credit card transactions using a Logistic Regression model. The dataset used for this project is sourced from Kaggle and contains anonymized transaction data with labeled legitimate and fraudulent transactions. The project implements Exploratory Data Analysis (EDA), data preprocessing, oversampling using SMOTE, and model training and evaluation.

## Dataset
- **File:** `creditcard.csv`
- **Attributes:** The dataset includes transaction features (`V1` to `V28`), the `Amount` column, and the `Class` column where:
  - `0`: Legitimate transaction
  - `1`: Fraudulent transaction

## Prerequisites
### Libraries Used:
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `imblearn`

### Installation
Ensure you have the required libraries installed:
```bash
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn
```

## Steps

### 1. Load and Explore the Dataset
- Load the dataset using `pandas`.
- Perform basic exploratory data analysis (EDA):
  - Check the dataset’s structure using `.head()`, `.info()`, and `.describe()`.
  - Verify class distribution with `value_counts()`.
  - Identify missing values using `.isna().sum()`.

### 2. Data Cleaning
- Remove duplicate rows using `.drop_duplicates()`.
- Recalculate the class distribution after deduplication.

### 3. Exploratory Data Analysis (EDA)
- Analyze transaction amounts for fraud and legitimate transactions.
- Visualize class distribution with a countplot.
- Generate and visualize a correlation matrix using a heatmap.
- Examine distributions of specific features (`Amount`, `V11`, `V2`, `V17`, `V4`, `V14`) with boxplots.

### 4. Data Preprocessing
- Normalize the `Amount` column using `StandardScaler`.
- Balance the dataset using SMOTE (Synthetic Minority Oversampling Technique).

### 5. Model Training
- Split the dataset into training and testing sets using an 80-20 split.
- Train a Logistic Regression model on the resampled dataset.

### 6. Model Evaluation
- Evaluate the model using the following metrics:
  - **Classification Report:** Precision, recall, F1-score.
  - **Confusion Matrix:** Visualized using a heatmap.
  - **Precision-Recall Curve.**
  - **ROC Curve:** Calculate and plot the Area Under Curve (AUC).

### 7. Hyperparameter Tuning
- Use GridSearchCV to find the optimal parameters for the Logistic Regression model.
- Best parameters: `C=10`, `penalty='l1'`.

## Results
- **Model Metrics:**
  - Accuracy: 97%
  - Precision, Recall, and F1-score: Achieved high scores for both classes.
- **ROC AUC Score:** Visualized to show the model's performance.

## Visualization Highlights
- **Class Distribution:** Shows the imbalance in the dataset.
- **Correlation Matrix:** Provides insight into feature relationships.
- **Boxplots:** Highlight outliers and feature distributions.
- **Precision-Recall Curve and ROC Curve:** Evaluate the trade-offs between precision and recall, and the true-positive rate versus false-positive rate, respectively.

## How to Run
1. Clone the repository and navigate to the project directory.
2. Install the required libraries using the provided commands.
3. Run the script in a Python environment with the `creditcard.csv` dataset in the `data` folder.

## Future Improvements
- Experiment with different machine learning models such as Random Forest or Gradient Boosting.
- Implement additional preprocessing techniques for better feature engineering.
- Use advanced hyperparameter optimization techniques like Bayesian Optimization.

## Acknowledgments
Dataset sourced from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).


