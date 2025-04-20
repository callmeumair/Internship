# Customer Churn Prediction Model

This project implements a machine learning model to predict customer churn for a business. It helps identify customers who are likely to stop using a service, enabling proactive retention measures.

## Project Structure
```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   └── app.py
├── data/
│   └── customer_data.csv
└── models/
    └── churn_model.joblib
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```bash
python src/data_preprocessing.py
```

2. Train Model:
```bash
python src/model.py
```

3. Evaluate Model:
```bash
python src/evaluate.py
```

4. Run Flask Application:
```bash
python src/app.py
```

## Model Details

The project uses a Random Forest Classifier to predict customer churn based on various features such as:
- Customer demographics
- Usage patterns
- Payment history
- Customer service interactions

## API Endpoints

- POST `/predict`: Accepts customer data and returns churn prediction
- GET `/health`: Health check endpoint

## Performance Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

## Author

Umer Shakilahmed Patel (EC2431201010074)
SRM Institute of Science and Technology 