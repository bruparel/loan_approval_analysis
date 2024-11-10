
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

def preprocess_data(data):
    """Prepare data for modeling."""
    X = data.drop(columns=['Loan_Status'])
    y = data['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    categorical_features = ['Gender', 'Married', 'Dependents', 'Graduate', 'Self_Employed', 'Credit_History', 'Property_Area']
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features
