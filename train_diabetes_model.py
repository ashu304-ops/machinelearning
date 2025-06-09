# train_diabetes_model.py
import joblib
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load a sample dataset (substitute for Pima if not available)
data = load_diabetes(as_frame=True)
X = data.data[['bmi', 'bp', 's1', 's5']]  # Just a few features for simplicity
y = (data.target > 100).astype(int)  # Convert to binary classification

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')
