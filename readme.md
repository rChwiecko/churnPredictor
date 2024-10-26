# Customer Churn Predictor Models

This repository contains machine learning models trained to predict customer churn. **Churn prediction** helps businesses identify customers who are likely to cancel their service, allowing for proactive engagement to retain them. Each model is serialized as a `.pkl` file, making it easy to load and use for predictions.

## Models Included

The following models are included in this repository:

- **XGB_model.pkl**: XGBoost model for churn prediction.
- **dt_model.pkl**: Decision Tree model for churn prediction.
- **knn_model.pkl**: K-Nearest Neighbors model for churn prediction.
- **nb_model.pkl**: Naive Bayes model for churn prediction.
- **rf_model.pkl**: Random Forest model for churn prediction.
- **svc_model.pkl**: Support Vector Classifier model for churn prediction.
- **voting-clf.pkl**: Ensemble Voting Classifier (soft voting) combining multiple models for a more robust prediction.
- **voting-clf-hard.pkl**: Ensemble Voting Classifier (hard voting).
- **xgb-feature-eng.pkl**: XGBoost model with feature engineering applied.
- **xgb-smote.pkl**: XGBoost model trained using SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.

Additional files:
- **model.ipynb**: Jupyter notebook containing model training, evaluation, and other analyses.
- **notes.txt**: Notes and additional information about model tuning, feature selection, etc.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.6 or higher
- Required libraries: `pickle`, `scikit-learn`, `pandas`, `xgboost`, `joblib`

You can install any missing libraries with the following command:
```bash
pip install scikit-learn pandas xgboost joblib
```
To get started, clone the repo to your local machine:

```bash
git clone https://github.com/rChwiecko/churn-predictor-models.git
```

## Loading User Models
Each .pkl file is a serialized model that can be loaded in Python for predictions. Here’s an example of how to load and use one of the models:

```python
import pickle
import pandas as pd

# Load your data (replace this with your actual data loading code)
data = pd.read_csv('./data/churn.csv')

# Load a model, e.g., knn_model
with open('knn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make a prediction
predictions = model.predict(data)
print(predictions)
```

## Example with ensemble classifier
To use the ensemble Voting Classifier, load it similarly and run predictions.
```python
with open('voting-clf.pkl', 'rb') as file:
    voting_model = pickle.load(file)

# Make predictions
voting_predictions = voting_model.predict(data)
print(voting_predictions)
```