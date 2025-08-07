# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Optional: set MLflow tracking URI (for remote server use)
# mlflow.set_tracking_uri("http://your-mlflow-server:5000")

# Load dataset
data = pd.read_csv('data/flights_full_featured.csv')  # Adjust the path as needed

# Feature engineering (example)
X = data.drop(columns=['target'])  # Replace 'target' with your actual target column
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():

    # Define and train model
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log model and metrics to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(clf, "model")

    print(f"Accuracy: {accuracy:.4f}")
