
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

def evaluate_models(models, X_test, y_test):
    """Evaluate multiple models and return their metrics in a dictionary."""
    evaluation_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        evaluation_results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Confusion Matrix": conf_matrix,
            "Classification Report": report
        }
    return evaluation_results
