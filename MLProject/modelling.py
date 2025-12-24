import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Data
print("Loading data...")
df = pd.read_csv('diabetes_prediction_dataset_preprocessing.csv')
X = df.drop(columns=['diabetes'])
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
print("Training model...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Evaluasi
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# Logging MLflow (Local)

print("Logging to MLflow...")
with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(rf, "model")
    
    mlflow.sklearn.save_model(rf, "model_output")

print("Training Selesai. Model saved to 'model_output'.")