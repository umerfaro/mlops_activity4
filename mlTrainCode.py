from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

# Set the MLflow experiment
mlflow.set_experiment('wine_rf_experiment')

# Load the Wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Convert to pandas DataFrame for better handling
feature_df = pd.DataFrame(X, columns=wine.feature_names)
target_df = pd.Series(y, name='target')

# Split the data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    feature_df, target_df, test_size=0.2, random_state=42
)

# Start an MLflow run
with mlflow.start_run(run_name="rf_with_grid_search"):
    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Base model
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search to find best hyperparameters
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    
    # Cross-validation on training data
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
    
    # Train the final model with best parameters
    best_rf.fit(X_train, y_train)
    
    # Make predictions and calculate metrics
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"Best Parameters: {grid_search.best_params_}")
    print(
        f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f} "
        f"± {np.std(cv_scores):.4f}"
    )
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))
    
    # Log all parameters, metrics to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("cv_accuracy_mean", np.mean(cv_scores))
    mlflow.log_metric("cv_accuracy_std", np.std(cv_scores))
    mlflow.log_metric("test_accuracy", accuracy)
    
    # Log feature importance
    feature_importances = pd.Series(
        best_rf.feature_importances_, 
        index=wine.feature_names
    ).sort_values(ascending=False)
    
    for feature, importance in feature_importances.items():
        mlflow.log_metric(f"importance_{feature}", importance)
    
    # Create a sample input for model signature
    input_example = X_train.iloc[:5]
    
    # Log the trained model with signature
    mlflow.sklearn.log_model(
        best_rf, 
        "random_forest_model",
        input_example=input_example
    )
    
    print("\nModel training completed and logged to MLflow.") 