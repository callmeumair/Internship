import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data():
    """
    Load the preprocessed test data
    """
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    return X_test, y_test

def load_model(filepath='models/churn_model.joblib'):
    """
    Load the trained model from disk
    """
    return joblib.load(filepath)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using various metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('data/confusion_matrix.png')
    plt.close()

def main():
    # Load test data and model
    X_test, y_test = load_test_data()
    model = load_model()
    
    # Evaluate model
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Log metrics
    logger.info("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric.capitalize()}: {value:.3f}")
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion matrix plot saved as 'data/confusion_matrix.png'")

if __name__ == "__main__":
    main() 