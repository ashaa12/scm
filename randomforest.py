import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("scm.csv")

# Remove rows missing target
df = df.dropna(subset=["WeekdayOrder"])

# Encode categorical columns
label_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in label_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# Split Data
X = df.drop("WeekdayOrder", axis=1)
y = df["WeekdayOrder"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Evaluation
pred = rf.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE:", mse)
print("R2 Score:", r2)

# Save model
joblib.dump(rf, "random_forest_model.joblib")
print("Model Saved!")
