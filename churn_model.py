import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Paths
DATA_PATH = 'data/churn_data.csv'
MODEL_PATH = 'models/churn_model.pkl'

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Drop customerID
df.drop(['customerID'], axis=1, inplace=True)

# Clean TotalCharges (convert to numeric, coerce errors)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values (e.g., TotalCharges)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode target: 'Yes' -> 1, 'No' -> 0
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Define features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Specify categorical and numerical columns
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod']

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Preprocessor: OneHotEncode categoricals, passthrough numericals
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

# Pipeline: preprocessing + RandomForest classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Evaluate (optional)
from sklearn.metrics import classification_report, confusion_matrix

y_pred = pipeline.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save pipeline
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"Model pipeline saved to {MODEL_PATH}")

