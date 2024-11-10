
import numpy as np

def get_logistic_equation(pipeline, numerical_features, categorical_features):
    """Retrieve logistic regression coefficients and intercept for the equation."""
    logistic_model = pipeline.named_steps['classifier']
    preprocessor_transformer = pipeline.named_steps['preprocessor']
    
    feature_names = numerical_features + list(preprocessor_transformer.named_transformers_['cat'].get_feature_names_out(categorical_features))
    coefficients = logistic_model.coef_.flatten()
    intercept = logistic_model.intercept_[0]

    equation = {"Intercept": intercept}
    equation.update(dict(zip(feature_names, coefficients)))
    return equation

def predict_approval(pipeline, applicant_data):
    """Predict loan approval for a new applicant based on the model."""
    return pipeline.predict(applicant_data), pipeline.predict_proba(applicant_data)[:, 1]
