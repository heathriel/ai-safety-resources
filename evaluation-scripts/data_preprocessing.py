"""
data_preprocessing.py
---------------------
This script performs advanced data preprocessing, including:
- Feature engineering
- Outlier detection and handling
- Data normalization
- Balancing imbalanced datasets using SMOTE
The processed dataset is saved for subsequent AI model training.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Step 1: Load Dataset
# Replace 'dataset.csv' with the path to your dataset
try:
    data = pd.read_csv('dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: dataset.csv not found. Please check the file path.")
    exit()

# Step 2: Feature Engineering
# Example: Create a new feature as the ratio of two existing features
if 'feature1' in data.columns and 'feature2' in data.columns:
    data['feature_ratio'] = data['feature1'] / data['feature2']
    print("Feature engineering completed: 'feature_ratio' created.")
else:
    print("Warning: 'feature1' or 'feature2' not found in the dataset. Skipping feature engineering.")

# Step 3: Outlier Detection and Handling
# Example: Identify outliers in 'feature1' above the 99th percentile
if 'feature1' in data.columns:
    outliers = data[data['feature1'] > data['feature1'].quantile(0.99)]
    print(f"Detected {len(outliers)} outliers in 'feature1'.")
    # Handle outliers: Cap values at the 99th percentile
    cap_value = data['feature1'].quantile(0.99)
    data['feature1'] = data['feature1'].apply(lambda x: min(x, cap_value))
    print("Outliers capped at the 99th percentile.")
else:
    print("Warning: 'feature1' not found in the dataset. Skipping outlier detection.")

# Step 4: Data Normalization
# Normalize numerical features for model training
numerical_features = ['feature1', 'feature2']  # Replace with relevant feature names
for feature in numerical_features:
    if feature in data.columns:
        scaler = StandardScaler()
        data[[feature]] = scaler.fit_transform(data[[feature]])
        print(f"Feature '{feature}' normalized.")
    else:
        print(f"Warning: '{feature}' not found in the dataset. Skipping normalization.")

# Step 5: Handle Imbalanced Datasets
# Apply SMOTE to balance the target variable
if 'target' in data.columns:
    X, y = data.drop('target', axis=1), data['target']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Imbalanced dataset handled using SMOTE.")
else:
    print("Error: 'target' column not found in the dataset. Unable to perform SMOTE.")
    exit()

# Step 6: Save Preprocessed Data
# Save the preprocessed dataset to a new CSV file
try:
    preprocessed_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='target')], axis=1)
    preprocessed_data.to_csv('preprocessed_dataset.csv', index=False)
    print("Preprocessed dataset saved as 'preprocessed_dataset.csv'.")
except Exception as e:
    print(f"Error saving preprocessed dataset: {e}")