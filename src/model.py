import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data():
    """
    Load the preprocessed training data
    """
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    return X_train, y_train

def train_model(X_train, y_train):
    """
    Train a Random Forest model with hyperparameter tuning
    """
    logger.info("Starting model training...")
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize the base model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1',
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def save_model(model, filepath='models/churn_model.joblib'):
    """
    Save the trained model to disk
    """
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

def main():
    # Load training data
    X_train, y_train = load_training_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Save model
    save_model(model)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 most important features:")
    logger.info(feature_importance.head(10))

if __name__ == "__main__":
    main() 