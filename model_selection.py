
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

def train_models(X_train, y_train, preprocessor):
    """Train multiple models and return them in a dictionary."""
    models = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ]),
        'Decision Tree': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'SVM': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(probability=True, random_state=42))
        ])
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models
