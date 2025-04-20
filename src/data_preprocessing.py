import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """
    Load sample customer data or create synthetic data for demonstration
    """
    # Creating synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(45, 15, n_samples),
        'tenure': np.random.randint(0, 72, n_samples),
        'monthly_charges': np.random.normal(70, 30, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])  # 27% churn rate
    }
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    and scaling numerical features
    """
    logger.info("Starting data preprocessing...")
    
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = df_processed.fillna({
        'total_charges': df_processed['monthly_charges'] * df_processed['tenure']
    })
    
    # Convert categorical variables to dummy variables
    categorical_columns = ['gender', 'internet_service', 'contract', 'payment_method']
    df_processed = pd.get_dummies(df_processed, columns=categorical_columns)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['age', 'tenure', 'monthly_charges', 'total_charges']
    df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
    
    logger.info("Data preprocessing completed.")
    return df_processed, scaler

def split_data(df, target_column='churn', test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def main():
    # Load data
    logger.info("Loading data...")
    df = load_data()
    
    # Save raw data
    df.to_csv('data/customer_data.csv', index=False)
    logger.info("Raw data saved to data/customer_data.csv")
    
    # Preprocess data
    df_processed, scaler = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    # Save processed data
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    
    logger.info("Preprocessed data saved successfully.")

if __name__ == "__main__":
    main() 