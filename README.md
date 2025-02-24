import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Feature Engineering
def feature_engineering(data):
    # Example feature engineering (add your own based on RTL analysis)
    data['fan_in_out_ratio'] = data['fan_in'] / (data['fan_out'] + 1)
    data['gate_complexity'] = data['and_gates'] + data['or_gates'] + data['not_gates']
    data['path_depth'] = data['longest_path']
    return data

# Prepare data for training
def prepare_data(data):
    features = ['fan_in', 'fan_out', 'fan_in_out_ratio', 'gate_complexity', 'path_depth']
    X = data[features]
    y = data['combinational_depth']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
    print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, predictions)))
    print("R² Score:", r2_score(y_test, predictions))

# Main function
if __name__ == '__main__':
    # Load and preprocess data
    data = load_data('rtl_data.csv')
    data = feature_engineering(data)
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Example prediction
    sample_input = np.array([[3, 5, 0.6, 10, 15]])
    predicted_depth = model.predict(sample_input)
    print("Predicted Combinational Depth:", predicted_depth)
