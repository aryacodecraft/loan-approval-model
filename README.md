# Loan Approval Predictor

## Overview
The Loan Approval Predictor is a machine learning project that aims to predict whether a loan application will be approved or not based on various features related to the applicants. This project utilizes a Decision Tree Classifier for its implementation, which is known for its simplicity and interpretability.

## Decision Tree Classifier
The Decision Tree Classifier is chosen for this project due to several advantages:

1. **Interpretability**: Decision trees provide a clear visual representation of the decision-making process, making it easier to understand how predictions are made.
2. **Non-Linear Relationships**: They can capture non-linear relationships between features without requiring transformation.
3. **Feature Importance**: Decision trees can easily identify the most important features influencing the predictions.
4. **No Need for Feature Scaling**: Unlike some other algorithms, decision trees do not require normalization or standardization of features.
5. **Robustness to Outliers**: Decision trees are less sensitive to outliers compared to models like linear regression.

While other models such as logistic regression, support vector machines, or ensemble methods like random forests may offer better accuracy in some cases, the Decision Tree Classifier strikes a balance between performance and interpretability, making it a suitable choice for this project.

## Dataset
The dataset used in this project is the "Loan Prediction Problem Dataset" by [altruistdelhite04](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) from Kaggle. It contains various features related to loan applicants, including:

- Applicant's income
- Co-applicant's income
- Loan amount
- Loan term
- Credit history
- Property area
- And more...

The target variable is `Loan_Status`, which indicates whether the loan was approved (1) or not (0).

## Setup Instructions
To set up and run the project, follow these steps:

1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd loan-approval-predictor
   ```

2. **Install Dependencies**:
   It is recommended to create a virtual environment before installing the dependencies. You can use `venv` or `conda` for this purpose.
   ```
   pip install -r requirements.txt
   ```

3. **Run the Predictor**:
   Execute the predictor script to train the model and evaluate its accuracy.
   ```
   python src/predictor.py
   ```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- The contributors of the Decision Tree Classifier implementation in scikit-learn.